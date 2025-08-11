#!/usr/bin/env python3
"""
Enhanced ML Performance Tracker

This module provides comprehensive ML model performance tracking with detailed
prediction analysis, ensemble performance monitoring, and model comparison.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


class ModelType(Enum):
    """ML model types."""

    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"


class PredictionType(Enum):
    """Prediction types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PROBABILITY = "probability"


@dataclass
class ModelPredictionRecord:
    """Individual model prediction record."""

    prediction_id: str
    model_id: str
    model_type: ModelType
    ensemble_name: str
    timestamp: datetime

    # Input features
    features: dict[str, float]
    feature_count: int

    # Prediction
    prediction: float
    confidence: float
    prediction_type: PredictionType
    probability_distribution: dict[str, float] = field(default_factory=dict)

    # Feature analysis
    feature_importance: dict[str, float] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)

    # Model metadata
    model_version: str = ""
    training_date: datetime | None = None
    last_retrain_date: datetime | None = None

    # Prediction context
    market_regime: str = ""
    symbol: str = ""
    timeframe: str = ""

    # Actual outcome (filled later)
    actual_outcome: float | None = None
    outcome_timestamp: datetime | None = None

    # Performance metrics (calculated later)
    prediction_error: float | None = None
    absolute_error: float | None = None
    squared_error: float | None = None
    directional_accuracy: bool | None = None


@dataclass
class EnsemblePerformanceRecord:
    """Ensemble performance tracking."""

    ensemble_id: str
    ensemble_name: str
    timestamp: datetime

    # Individual model predictions
    individual_predictions: list[ModelPredictionRecord]

    # Ensemble aggregation
    aggregation_method: str  # "weighted_average", "voting", "meta_learner"
    final_prediction: float
    ensemble_confidence: float

    # Consensus metrics
    prediction_variance: float
    consensus_level: float  # How much models agree
    disagreement_score: float
    outlier_models: list[str] = field(default_factory=list)

    # Meta-learner details (if applicable)
    meta_learner_prediction: float | None = None
    meta_learner_confidence: float | None = None
    meta_learner_features: dict[str, float] = field(default_factory=dict)

    # Performance (filled later)
    actual_outcome: float | None = None
    ensemble_error: float | None = None
    best_individual_error: float | None = None
    worst_individual_error: float | None = None
    ensemble_improvement: float | None = None  # vs best individual


@dataclass
class ModelPerformanceAnalysis:
    """Comprehensive model performance analysis."""

    model_id: str
    model_type: ModelType
    analysis_period_days: int
    timestamp: datetime

    # Prediction statistics
    total_predictions: int
    successful_predictions: int
    failed_predictions: int

    # Accuracy metrics
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_squared_error: float
    r_squared: float | None = None

    # Classification metrics (if applicable)
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_score: float | None = None

    # Directional accuracy
    directional_accuracy: float
    up_prediction_accuracy: float
    down_prediction_accuracy: float

    # Confidence analysis
    confidence_calibration: float  # How well confidence matches actual accuracy
    overconfidence_score: float
    underconfidence_score: float

    # Feature analysis
    feature_stability_score: float
    most_important_features: list[str]
    feature_drift_score: float

    # Temporal performance
    performance_trend: str  # "improving", "declining", "stable"
    recent_performance_change: float
    performance_volatility: float

    # Comparison metrics
    relative_performance_rank: int | None = None
    performance_percentile: float | None = None

    # Regime-specific performance
    regime_performance: dict[str, float] = field(default_factory=dict)
    best_performing_regime: str = ""
    worst_performing_regime: str = ""


@dataclass
class ModelComparisonReport:
    """Model comparison and ranking report."""

    comparison_id: str
    timestamp: datetime
    comparison_period_days: int

    # Models included
    models_analyzed: list[str]
    ensemble_models: list[str]

    # Overall rankings
    performance_ranking: list[tuple[str, float]]  # (model_id, score)
    stability_ranking: list[tuple[str, float]]
    efficiency_ranking: list[tuple[str, float]]

    # Best performers by metric
    best_accuracy: tuple[str, float]
    best_precision: tuple[str, float]
    best_recall: tuple[str, float]
    best_f1: tuple[str, float]
    most_stable: tuple[str, float]
    most_consistent: tuple[str, float]

    # Ensemble analysis
    ensemble_effectiveness: dict[
        str,
        float,
    ]  # How much ensemble improves over individuals
    best_ensemble_combination: list[str]
    ensemble_diversity_score: float

    # Regime-specific analysis
    regime_specialists: dict[str, str]  # regime -> best model
    regime_generalists: list[str]  # Models good across regimes

    # Recommendations
    model_recommendations: list[str]
    ensemble_recommendations: list[str]
    retraining_recommendations: list[str]


class EnhancedMLTracker:
    """
    Enhanced ML performance tracker with comprehensive prediction analysis,
    ensemble monitoring, and model comparison capabilities.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize enhanced ML tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedMLTracker")

        # Configuration
        self.tracker_config = config.get("enhanced_ml_tracker", {})
        self.enable_real_time_tracking = self.tracker_config.get(
            "enable_real_time_tracking",
            True,
        )
        self.enable_ensemble_analysis = self.tracker_config.get(
            "enable_ensemble_analysis",
            True,
        )
        self.enable_model_comparison = self.tracker_config.get(
            "enable_model_comparison",
            True,
        )
        self.performance_window_days = self.tracker_config.get(
            "performance_window_days",
            7,
        )
        self.min_predictions_for_analysis = self.tracker_config.get(
            "min_predictions_for_analysis",
            50,
        )

        # Storage
        self.prediction_records: dict[str, ModelPredictionRecord] = {}
        self.ensemble_records: dict[str, EnsemblePerformanceRecord] = {}
        self.performance_analyses: dict[str, ModelPerformanceAnalysis] = {}
        self.comparison_reports: list[ModelComparisonReport] = []

        # Real-time tracking
        self.active_predictions: dict[str, datetime] = {}  # prediction_id -> timestamp
        self.pending_outcomes: dict[str, ModelPredictionRecord] = {}

        # Performance caches
        self.model_performance_cache: dict[str, dict[str, float]] = {}
        self.ensemble_performance_cache: dict[str, dict[str, float]] = {}

        # Statistics
        self.tracking_stats = {
            "total_predictions_tracked": 0,
            "ensembles_tracked": 0,
            "models_analyzed": 0,
            "comparisons_generated": 0,
            "last_update": datetime.now(),
        }

        self.is_initialized = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML tracker configuration"),
            AttributeError: (False, "Missing required tracker parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ML tracker initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the enhanced ML tracker.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Enhanced ML Tracker...")

            # Initialize storage backend if needed
            await self._initialize_storage()

            # Load historical data
            await self._load_historical_data()

            # Initialize performance analysis
            await self._initialize_performance_analysis()

            # Start background tasks
            await self._start_background_tasks()

            self.is_initialized = True
            self.logger.info("✅ Enhanced ML Tracker initialized successfully")
            return True

        except Exception as e:
            self.print(failed("❌ Enhanced ML Tracker initialization failed: {e}"))
            return False

    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        try:
            # Use the same storage backend as other monitoring components
            storage_backend = self.config.get("monitoring", {}).get(
                "storage_backend",
                "sqlite",
            )

            if storage_backend == "sqlite":
                from src.database.sqlite_manager import SQLiteManager

                self.storage_manager = SQLiteManager(self.config)
                await self.storage_manager.initialize()
                await self._create_ml_tracking_tables()

            self.logger.info(
                f"ML tracker storage backend '{storage_backend}' initialized",
            )

        except Exception as e:
            self.print(failed("Failed to initialize ML tracker storage: {e}"))
            raise

    async def _create_ml_tracking_tables(self) -> None:
        """Create database tables for ML tracking."""
        try:
            tables = [
                """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    model_type TEXT,
                    ensemble_name TEXT,
                    timestamp DATETIME,
                    prediction REAL,
                    confidence REAL,
                    actual_outcome REAL,
                    prediction_error REAL,
                    features_json TEXT,
                    feature_importance_json TEXT,
                    prediction_metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS ensemble_performance (
                    ensemble_id TEXT PRIMARY KEY,
                    ensemble_name TEXT,
                    timestamp DATETIME,
                    final_prediction REAL,
                    ensemble_confidence REAL,
                    consensus_level REAL,
                    actual_outcome REAL,
                    ensemble_error REAL,
                    individual_predictions_json TEXT,
                    performance_metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS model_performance_analysis (
                    analysis_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    model_type TEXT,
                    analysis_period_days INTEGER,
                    timestamp DATETIME,
                    total_predictions INTEGER,
                    mean_absolute_error REAL,
                    directional_accuracy REAL,
                    confidence_calibration REAL,
                    performance_trend TEXT,
                    analysis_details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    comparison_period_days INTEGER,
                    models_count INTEGER,
                    best_model TEXT,
                    comparison_results TEXT,
                    recommendations TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ]

            for table_sql in tables:
                await self.storage_manager.execute_query(table_sql)

            self.logger.info("ML tracking tables created successfully")

        except Exception as e:
            self.print(failed("Failed to create ML tracking tables: {e}"))
            raise

    async def _load_historical_data(self) -> None:
        """Load historical ML tracking data."""
        try:
            # Load recent prediction records for analysis
            cutoff_date = datetime.now() - timedelta(days=self.performance_window_days)

            if hasattr(self, "storage_manager"):
                query = """
                SELECT * FROM ml_predictions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                """

                results = await self.storage_manager.execute_query(
                    query,
                    (cutoff_date,),
                )

                for _row in results:
                    # Reconstruct prediction record from database
                    # This is a simplified version - full implementation would deserialize JSON
                    pass

            self.logger.info("Historical ML tracking data loaded")

        except Exception as e:
            self.print(failed("Failed to load historical ML data: {e}"))
            # Non-critical error, continue

    async def _initialize_performance_analysis(self) -> None:
        """Initialize performance analysis components."""
        try:
            # Initialize performance caches
            self.model_performance_cache.clear()
            self.ensemble_performance_cache.clear()

            self.logger.info("Performance analysis components initialized")

        except Exception as e:
            self.print(failed("Failed to initialize performance analysis: {e}"))
            raise

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Start periodic performance analysis
            if self.enable_real_time_tracking:
                asyncio.create_task(self._periodic_performance_analysis())

            # Start model comparison task
            if self.enable_model_comparison:
                asyncio.create_task(self._periodic_model_comparison())

            self.logger.info("Background ML tracking tasks started")

        except Exception as e:
            self.print(failed("Failed to start background tasks: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model prediction tracking",
    )
    async def track_model_prediction(
        self,
        model_id: str,
        model_type: ModelType,
        ensemble_name: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        feature_importance: dict[str, float] = None,
        **kwargs,
    ) -> str:
        """
        Track an individual model prediction.

        Args:
            model_id: Unique model identifier
            model_type: Type of ML model
            ensemble_name: Name of the ensemble this model belongs to
            prediction: Model prediction value
            confidence: Prediction confidence
            features: Input features used for prediction
            feature_importance: Feature importance scores
            **kwargs: Additional metadata

        Returns:
            str: Prediction ID for tracking
        """
        try:
            prediction_id = f"{model_id}_{int(time.time() * 1000)}"

            # Create prediction record
            record = ModelPredictionRecord(
                prediction_id=prediction_id,
                model_id=model_id,
                model_type=model_type,
                ensemble_name=ensemble_name,
                timestamp=datetime.now(),
                features=features,
                feature_count=len(features),
                prediction=prediction,
                confidence=confidence,
                prediction_type=kwargs.get(
                    "prediction_type",
                    PredictionType.REGRESSION,
                ),
                probability_distribution=kwargs.get("probability_distribution", {}),
                feature_importance=feature_importance or {},
                top_features=list((feature_importance or {}).keys())[:10],
                model_version=kwargs.get("model_version", ""),
                training_date=kwargs.get("training_date"),
                last_retrain_date=kwargs.get("last_retrain_date"),
                market_regime=kwargs.get("market_regime", ""),
                symbol=kwargs.get("symbol", ""),
                timeframe=kwargs.get("timeframe", ""),
            )

            # Store record
            self.prediction_records[prediction_id] = record
            self.pending_outcomes[prediction_id] = record

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_prediction_record(record)

            # Update statistics
            self.tracking_stats["total_predictions_tracked"] += 1
            self.tracking_stats["last_update"] = datetime.now()

            self.logger.debug(
                f"Tracked prediction {prediction_id} for model {model_id}",
            )

            return prediction_id

        except Exception as e:
            self.print(failed("Failed to track model prediction: {e}"))
            return ""

    async def _store_prediction_record(self, record: ModelPredictionRecord) -> None:
        """Store prediction record in database."""
        try:
            data = {
                "prediction_id": record.prediction_id,
                "model_id": record.model_id,
                "model_type": record.model_type.value,
                "ensemble_name": record.ensemble_name,
                "timestamp": record.timestamp,
                "prediction": record.prediction,
                "confidence": record.confidence,
                "actual_outcome": record.actual_outcome,
                "prediction_error": record.prediction_error,
                "features_json": json.dumps(record.features),
                "feature_importance_json": json.dumps(record.feature_importance),
                "prediction_metadata": json.dumps(asdict(record), default=str),
            }

            await self.storage_manager.insert_data("ml_predictions", data)

        except Exception as e:
            self.print(failed("Failed to store prediction record: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ensemble performance tracking",
    )
    async def track_ensemble_performance(
        self,
        ensemble_name: str,
        individual_predictions: list[ModelPredictionRecord],
        final_prediction: float,
        ensemble_confidence: float,
        aggregation_method: str = "weighted_average",
        **kwargs,
    ) -> str:
        """
        Track ensemble performance.

        Args:
            ensemble_name: Name of the ensemble
            individual_predictions: List of individual model predictions
            final_prediction: Final ensemble prediction
            ensemble_confidence: Ensemble confidence
            aggregation_method: Method used to aggregate predictions
            **kwargs: Additional metadata

        Returns:
            str: Ensemble record ID
        """
        try:
            ensemble_id = f"{ensemble_name}_{int(time.time() * 1000)}"

            # Calculate consensus metrics
            predictions = [p.prediction for p in individual_predictions]
            prediction_variance = float(np.var(predictions)) if predictions else 0.0

            # Simple consensus level calculation
            mean_pred = np.mean(predictions) if predictions else 0.0
            consensus_level = 1.0 - (prediction_variance / (mean_pred**2 + 1e-8))
            disagreement_score = prediction_variance

            # Identify outlier models
            outlier_models = []
            if len(predictions) > 2:
                q75, q25 = np.percentile(predictions, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr

                for pred_record in individual_predictions:
                    if (
                        pred_record.prediction < lower_bound
                        or pred_record.prediction > upper_bound
                    ):
                        outlier_models.append(pred_record.model_id)

            # Create ensemble record
            record = EnsemblePerformanceRecord(
                ensemble_id=ensemble_id,
                ensemble_name=ensemble_name,
                timestamp=datetime.now(),
                individual_predictions=individual_predictions,
                aggregation_method=aggregation_method,
                final_prediction=final_prediction,
                ensemble_confidence=ensemble_confidence,
                prediction_variance=prediction_variance,
                consensus_level=consensus_level,
                disagreement_score=disagreement_score,
                outlier_models=outlier_models,
                meta_learner_prediction=kwargs.get("meta_learner_prediction"),
                meta_learner_confidence=kwargs.get("meta_learner_confidence"),
                meta_learner_features=kwargs.get("meta_learner_features", {}),
            )

            # Store record
            self.ensemble_records[ensemble_id] = record

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_ensemble_record(record)

            # Update statistics
            self.tracking_stats["ensembles_tracked"] += 1
            self.tracking_stats["last_update"] = datetime.now()

            self.logger.debug(f"Tracked ensemble performance {ensemble_id}")

            return ensemble_id

        except Exception as e:
            self.print(failed("Failed to track ensemble performance: {e}"))
            return ""

    async def _store_ensemble_record(self, record: EnsemblePerformanceRecord) -> None:
        """Store ensemble record in database."""
        try:
            data = {
                "ensemble_id": record.ensemble_id,
                "ensemble_name": record.ensemble_name,
                "timestamp": record.timestamp,
                "final_prediction": record.final_prediction,
                "ensemble_confidence": record.ensemble_confidence,
                "consensus_level": record.consensus_level,
                "actual_outcome": record.actual_outcome,
                "ensemble_error": record.ensemble_error,
                "individual_predictions_json": json.dumps(
                    [asdict(p, default=str) for p in record.individual_predictions],
                ),
                "performance_metadata": json.dumps(asdict(record), default=str),
            }

            await self.storage_manager.insert_data("ensemble_performance", data)

        except Exception as e:
            self.print(failed("Failed to store ensemble record: {e}"))
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="outcome recording",
    )
    async def record_actual_outcome(
        self,
        prediction_id: str,
        actual_outcome: float,
        outcome_timestamp: datetime = None,
    ) -> bool:
        """
        Record the actual outcome for a prediction.

        Args:
            prediction_id: ID of the prediction
            actual_outcome: Actual outcome value
            outcome_timestamp: When the outcome was observed

        Returns:
            bool: True if recorded successfully
        """
        try:
            if outcome_timestamp is None:
                outcome_timestamp = datetime.now()

            # Update prediction record
            if prediction_id in self.prediction_records:
                record = self.prediction_records[prediction_id]
                record.actual_outcome = actual_outcome
                record.outcome_timestamp = outcome_timestamp

                # Calculate performance metrics
                record.prediction_error = record.prediction - actual_outcome
                record.absolute_error = abs(record.prediction_error)
                record.squared_error = record.prediction_error**2

                # Directional accuracy
                predicted_direction = 1 if record.prediction > 0 else -1
                actual_direction = 1 if actual_outcome > 0 else -1
                record.directional_accuracy = predicted_direction == actual_direction

                # Remove from pending outcomes
                if prediction_id in self.pending_outcomes:
                    del self.pending_outcomes[prediction_id]

                # Update database
                if hasattr(self, "storage_manager"):
                    await self._update_prediction_outcome(record)

                self.logger.debug(f"Recorded outcome for prediction {prediction_id}")
                return True

            self.logger.warning(
                f"Prediction {prediction_id} not found for outcome recording",
            )
            return False

        except Exception as e:
            self.print(failed("Failed to record actual outcome: {e}"))
            return False

    async def _update_prediction_outcome(self, record: ModelPredictionRecord) -> None:
        """Update prediction record with outcome in database."""
        try:
            update_data = {
                "actual_outcome": record.actual_outcome,
                "prediction_error": record.prediction_error,
                "prediction_metadata": json.dumps(asdict(record), default=str),
            }

            await self.storage_manager.update_data(
                "ml_predictions",
                update_data,
                {"prediction_id": record.prediction_id},
            )

        except Exception as e:
            self.logger.exception(
                f"Failed to update prediction outcome in database: {e}",
            )
            raise

    async def generate_model_performance_analysis(
        self,
        model_id: str,
        analysis_period_days: int = None,
    ) -> ModelPerformanceAnalysis | None:
        """
        Generate comprehensive performance analysis for a model.

        Args:
            model_id: Model to analyze
            analysis_period_days: Analysis period (default: config value)

        Returns:
            ModelPerformanceAnalysis: Performance analysis or None if insufficient data
        """
        try:
            if analysis_period_days is None:
                analysis_period_days = self.performance_window_days

            # Get predictions for this model
            cutoff_date = datetime.now() - timedelta(days=analysis_period_days)

            model_predictions = [
                record
                for record in self.prediction_records.values()
                if (
                    record.model_id == model_id
                    and record.timestamp >= cutoff_date
                    and record.actual_outcome is not None
                )
            ]

            if len(model_predictions) < self.min_predictions_for_analysis:
                self.logger.warning(
                    f"Insufficient predictions for model {model_id} analysis",
                )
                return None

            # Calculate performance metrics
            [p.prediction for p in model_predictions]
            actuals = [p.actual_outcome for p in model_predictions]
            errors = [p.prediction_error for p in model_predictions]
            abs_errors = [abs(e) for e in errors]

            # Basic metrics
            mean_absolute_error = float(np.mean(abs_errors))
            root_mean_squared_error = float(np.sqrt(np.mean([e**2 for e in errors])))
            mean_squared_error = float(np.mean([e**2 for e in errors]))

            # R-squared (if applicable)
            r_squared = None
            if len(actuals) > 1:
                ss_res = sum([e**2 for e in errors])
                ss_tot = sum([(a - np.mean(actuals)) ** 2 for a in actuals])
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Directional accuracy
            directional_correct = sum(
                [1 for p in model_predictions if p.directional_accuracy],
            )
            directional_accuracy = directional_correct / len(model_predictions)

            # Up/down prediction accuracy
            up_predictions = [p for p in model_predictions if p.prediction > 0]
            down_predictions = [p for p in model_predictions if p.prediction <= 0]

            up_prediction_accuracy = (
                sum([1 for p in up_predictions if p.directional_accuracy])
                / len(up_predictions)
                if up_predictions
                else 0.0
            )
            down_prediction_accuracy = (
                sum([1 for p in down_predictions if p.directional_accuracy])
                / len(down_predictions)
                if down_predictions
                else 0.0
            )

            # Confidence calibration
            confidence_calibration = await self._calculate_confidence_calibration(
                model_predictions,
            )

            # Feature stability
            feature_stability_score = await self._calculate_feature_stability(
                model_predictions,
            )

            # Performance trend
            performance_trend, recent_change = await self._analyze_performance_trend(
                model_predictions,
            )

            # Create analysis
            analysis = ModelPerformanceAnalysis(
                model_id=model_id,
                model_type=model_predictions[0].model_type,
                analysis_period_days=analysis_period_days,
                timestamp=datetime.now(),
                total_predictions=len(model_predictions),
                successful_predictions=len(model_predictions),  # All have outcomes
                failed_predictions=0,
                mean_absolute_error=mean_absolute_error,
                root_mean_squared_error=root_mean_squared_error,
                mean_squared_error=mean_squared_error,
                r_squared=r_squared,
                directional_accuracy=directional_accuracy,
                up_prediction_accuracy=up_prediction_accuracy,
                down_prediction_accuracy=down_prediction_accuracy,
                confidence_calibration=confidence_calibration,
                overconfidence_score=max(
                    0,
                    confidence_calibration - directional_accuracy,
                ),
                underconfidence_score=max(
                    0,
                    directional_accuracy - confidence_calibration,
                ),
                feature_stability_score=feature_stability_score,
                most_important_features=await self._get_most_important_features(
                    model_predictions,
                ),
                feature_drift_score=0.0,  # Placeholder
                performance_trend=performance_trend,
                recent_performance_change=recent_change,
                performance_volatility=float(np.std(abs_errors)),
            )

            # Store analysis
            self.performance_analyses[model_id] = analysis

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_performance_analysis(analysis)

            self.logger.info(f"Generated performance analysis for model {model_id}")

            return analysis

        except Exception as e:
            self.logger.exception(
                f"Failed to generate performance analysis for model {model_id}: {e}",
            )
            return None

    async def _calculate_confidence_calibration(
        self,
        predictions: list[ModelPredictionRecord],
    ) -> float:
        """Calculate confidence calibration score."""
        try:
            if not predictions:
                return 0.0

            # Simple calibration: correlation between confidence and accuracy
            confidences = [p.confidence for p in predictions]
            accuracies = [1.0 if p.directional_accuracy else 0.0 for p in predictions]

            if len(set(confidences)) <= 1 or len(set(accuracies)) <= 1:
                return 0.5  # Default if no variance

            correlation = np.corrcoef(confidences, accuracies)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.5

        except Exception as e:
            self.print(failed("Failed to calculate confidence calibration: {e}"))
            return 0.5

    async def _calculate_feature_stability(
        self,
        predictions: list[ModelPredictionRecord],
    ) -> float:
        """Calculate feature importance stability."""
        try:
            if len(predictions) < 2:
                return 1.0

            # Calculate variance in feature importance rankings
            feature_rankings = []
            for pred in predictions:
                if pred.feature_importance:
                    # Convert to ranking
                    sorted_features = sorted(
                        pred.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    ranking = {
                        feat: idx for idx, (feat, _) in enumerate(sorted_features)
                    }
                    feature_rankings.append(ranking)

            if len(feature_rankings) < 2:
                return 1.0

            # Calculate stability as inverse of ranking variance
            all_features = set()
            for ranking in feature_rankings:
                all_features.update(ranking.keys())

            if not all_features:
                return 1.0

            stability_scores = []
            for feature in all_features:
                ranks = [
                    ranking.get(feature, len(all_features))
                    for ranking in feature_rankings
                ]
                variance = np.var(ranks)
                stability = 1.0 / (1.0 + variance)
                stability_scores.append(stability)

            return float(np.mean(stability_scores))

        except Exception as e:
            self.print(failed("Failed to calculate feature stability: {e}"))
            return 1.0

    async def _analyze_performance_trend(
        self,
        predictions: list[ModelPredictionRecord],
    ) -> tuple[str, float]:
        """Analyze performance trend over time."""
        try:
            if len(predictions) < 10:
                return "stable", 0.0

            # Sort by timestamp
            sorted_preds = sorted(predictions, key=lambda x: x.timestamp)

            # Calculate rolling accuracy
            window_size = max(10, len(sorted_preds) // 5)
            rolling_accuracies = []

            for i in range(window_size, len(sorted_preds)):
                window = sorted_preds[i - window_size : i]
                accuracy = sum([1 for p in window if p.directional_accuracy]) / len(
                    window,
                )
                rolling_accuracies.append(accuracy)

            if len(rolling_accuracies) < 2:
                return "stable", 0.0

            # Calculate trend
            x = np.arange(len(rolling_accuracies))
            y = np.array(rolling_accuracies)

            slope, _ = np.polyfit(x, y, 1)
            recent_change = rolling_accuracies[-1] - rolling_accuracies[0]

            if slope > 0.01:
                trend = "improving"
            elif slope < -0.01:
                trend = "declining"
            else:
                trend = "stable"

            return trend, float(recent_change)

        except Exception as e:
            self.print(failed("Failed to analyze performance trend: {e}"))
            return "stable", 0.0

    async def _get_most_important_features(
        self,
        predictions: list[ModelPredictionRecord],
    ) -> list[str]:
        """Get most consistently important features."""
        try:
            feature_importance_sum = {}
            feature_count = {}

            for pred in predictions:
                for feature, importance in pred.feature_importance.items():
                    feature_importance_sum[feature] = (
                        feature_importance_sum.get(feature, 0) + importance
                    )
                    feature_count[feature] = feature_count.get(feature, 0) + 1

            # Calculate average importance
            avg_importance = {}
            for feature in feature_importance_sum:
                avg_importance[feature] = (
                    feature_importance_sum[feature] / feature_count[feature]
                )

            # Sort by importance
            sorted_features = sorted(
                avg_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            return [feature for feature, _ in sorted_features[:10]]

        except Exception as e:
            self.print(failed("Failed to get most important features: {e}"))
            return []

    async def _store_performance_analysis(
        self,
        analysis: ModelPerformanceAnalysis,
    ) -> None:
        """Store performance analysis in database."""
        try:
            data = {
                "analysis_id": f"{analysis.model_id}_{int(analysis.timestamp.timestamp())}",
                "model_id": analysis.model_id,
                "model_type": analysis.model_type.value,
                "analysis_period_days": analysis.analysis_period_days,
                "timestamp": analysis.timestamp,
                "total_predictions": analysis.total_predictions,
                "mean_absolute_error": analysis.mean_absolute_error,
                "directional_accuracy": analysis.directional_accuracy,
                "confidence_calibration": analysis.confidence_calibration,
                "performance_trend": analysis.performance_trend,
                "analysis_details": json.dumps(asdict(analysis), default=str),
            }

            await self.storage_manager.insert_data("model_performance_analysis", data)

        except Exception as e:
            self.print(failed("Failed to store performance analysis: {e}"))
            raise

    async def _periodic_performance_analysis(self) -> None:
        """Periodic performance analysis task."""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour

                # Analyze all active models
                active_models = {
                    record.model_id for record in self.prediction_records.values()
                }

                for model_id in active_models:
                    await self.generate_model_performance_analysis(model_id)

                self.logger.debug("Completed periodic performance analysis")

        except asyncio.CancelledError:
            self.logger.info("Periodic performance analysis task cancelled")
        except Exception as e:
            self.print(error("Error in periodic performance analysis: {e}"))

    async def _periodic_model_comparison(self) -> None:
        """Periodic model comparison task."""
        try:
            while True:
                await asyncio.sleep(3600 * 6)  # Run every 6 hours

                # Generate model comparison report
                await self.generate_model_comparison_report()

                self.logger.debug("Completed periodic model comparison")

        except asyncio.CancelledError:
            self.logger.info("Periodic model comparison task cancelled")
        except Exception as e:
            self.print(error("Error in periodic model comparison: {e}"))

    async def generate_model_comparison_report(
        self,
        comparison_period_days: int = None,
    ) -> ModelComparisonReport | None:
        """
        Generate comprehensive model comparison report.

        Args:
            comparison_period_days: Comparison period (default: config value)

        Returns:
            ModelComparisonReport: Comparison report or None if insufficient data
        """
        try:
            if comparison_period_days is None:
                comparison_period_days = self.performance_window_days

            # Get all models with sufficient data
            cutoff_date = datetime.now() - timedelta(days=comparison_period_days)

            model_data = {}
            for record in self.prediction_records.values():
                if (
                    record.timestamp >= cutoff_date
                    and record.actual_outcome is not None
                ):
                    if record.model_id not in model_data:
                        model_data[record.model_id] = []
                    model_data[record.model_id].append(record)

            # Filter models with sufficient predictions
            qualified_models = {
                model_id: predictions
                for model_id, predictions in model_data.items()
                if len(predictions) >= self.min_predictions_for_analysis
            }

            if len(qualified_models) < 2:
                self.print(warning("Insufficient models for comparison"))
                return None

            # Calculate metrics for each model
            model_metrics = {}
            for model_id, predictions in qualified_models.items():
                metrics = await self._calculate_comparison_metrics(predictions)
                model_metrics[model_id] = metrics

            # Generate rankings
            performance_ranking = sorted(
                model_metrics.items(),
                key=lambda x: x[1]["overall_score"],
                reverse=True,
            )

            stability_ranking = sorted(
                model_metrics.items(),
                key=lambda x: x[1]["stability_score"],
                reverse=True,
            )

            # Find best performers
            best_accuracy = max(
                model_metrics.items(),
                key=lambda x: x[1]["directional_accuracy"],
            )

            # Create comparison report
            comparison_id = f"comparison_{int(time.time())}"

            report = ModelComparisonReport(
                comparison_id=comparison_id,
                timestamp=datetime.now(),
                comparison_period_days=comparison_period_days,
                models_analyzed=list(qualified_models.keys()),
                ensemble_models=[],  # Placeholder
                performance_ranking=[
                    (model, metrics["overall_score"])
                    for model, metrics in performance_ranking
                ],
                stability_ranking=[
                    (model, metrics["stability_score"])
                    for model, metrics in stability_ranking
                ],
                efficiency_ranking=[
                    (model, metrics["efficiency_score"])
                    for model, metrics in model_metrics.items()
                ],
                best_accuracy=best_accuracy,
                best_precision=(
                    best_accuracy[0],
                    best_accuracy[1]["directional_accuracy"],
                ),  # Simplified
                best_recall=(
                    best_accuracy[0],
                    best_accuracy[1]["directional_accuracy"],
                ),
                best_f1=(best_accuracy[0], best_accuracy[1]["directional_accuracy"]),
                most_stable=stability_ranking[0],
                most_consistent=stability_ranking[0],
                ensemble_effectiveness={},  # Placeholder
                best_ensemble_combination=[],
                ensemble_diversity_score=0.0,
                regime_specialists={},  # Placeholder
                regime_generalists=[],
                model_recommendations=await self._generate_model_recommendations(
                    model_metrics,
                ),
                ensemble_recommendations=[],
                retraining_recommendations=await self._generate_retraining_recommendations(
                    model_metrics,
                ),
            )

            # Store report
            self.comparison_reports.append(report)

            # Store in database
            if hasattr(self, "storage_manager"):
                await self._store_comparison_report(report)

            # Update statistics
            self.tracking_stats["comparisons_generated"] += 1
            self.tracking_stats["last_update"] = datetime.now()

            self.logger.info(f"Generated model comparison report {comparison_id}")

            return report

        except Exception as e:
            self.print(failed("Failed to generate model comparison report: {e}"))
            return None

    async def _calculate_comparison_metrics(
        self,
        predictions: list[ModelPredictionRecord],
    ) -> dict[str, float]:
        """Calculate comparison metrics for a model."""
        try:
            errors = [abs(p.prediction_error) for p in predictions]
            directional_correct = sum(
                [1 for p in predictions if p.directional_accuracy],
            )

            metrics = {
                "directional_accuracy": directional_correct / len(predictions),
                "mean_absolute_error": float(np.mean(errors)),
                "error_variance": float(np.var(errors)),
                "stability_score": 1.0 / (1.0 + np.var(errors)),
                "efficiency_score": directional_correct
                / len(predictions),  # Simplified
                "prediction_count": len(predictions),
            }

            # Overall score (weighted combination)
            metrics["overall_score"] = (
                0.4 * metrics["directional_accuracy"]
                + 0.3 * metrics["stability_score"]
                + 0.3 * metrics["efficiency_score"]
            )

            return metrics

        except Exception as e:
            self.print(failed("Failed to calculate comparison metrics: {e}"))
            return {}

    async def _generate_model_recommendations(
        self,
        model_metrics: dict[str, dict[str, float]],
    ) -> list[str]:
        """Generate model recommendations based on performance."""
        try:
            recommendations = []

            # Find best performing models
            best_models = sorted(
                model_metrics.items(),
                key=lambda x: x[1]["overall_score"],
                reverse=True,
            )[:3]

            for model_id, metrics in best_models:
                if metrics["overall_score"] > 0.7:
                    recommendations.append(
                        f"Increase allocation to {model_id} (score: {metrics['overall_score']:.2f})",
                    )

            # Find underperforming models
            worst_models = sorted(
                model_metrics.items(),
                key=lambda x: x[1]["overall_score"],
            )[:2]

            for model_id, metrics in worst_models:
                if metrics["overall_score"] < 0.4:
                    recommendations.append(
                        f"Consider reducing allocation to {model_id} (score: {metrics['overall_score']:.2f})",
                    )

            return recommendations

        except Exception as e:
            self.print(failed("Failed to generate model recommendations: {e}"))
            return []

    async def _generate_retraining_recommendations(
        self,
        model_metrics: dict[str, dict[str, float]],
    ) -> list[str]:
        """Generate retraining recommendations."""
        try:
            recommendations = []

            for model_id, metrics in model_metrics.items():
                if metrics["directional_accuracy"] < 0.5:
                    recommendations.append(f"Retrain {model_id} - accuracy below 50%")
                elif metrics["stability_score"] < 0.3:
                    recommendations.append(
                        f"Retrain {model_id} - high prediction variance",
                    )

            return recommendations

        except Exception as e:
            self.print(failed("Failed to generate retraining recommendations: {e}"))
            return []

    async def _store_comparison_report(self, report: ModelComparisonReport) -> None:
        """Store comparison report in database."""
        try:
            data = {
                "comparison_id": report.comparison_id,
                "timestamp": report.timestamp,
                "comparison_period_days": report.comparison_period_days,
                "models_count": len(report.models_analyzed),
                "best_model": report.performance_ranking[0][0]
                if report.performance_ranking
                else "",
                "comparison_results": json.dumps(asdict(report), default=str),
                "recommendations": json.dumps(report.model_recommendations),
            }

            await self.storage_manager.insert_data("model_comparisons", data)

        except Exception as e:
            self.print(failed("Failed to store comparison report: {e}"))
            raise

    async def get_tracking_statistics(self) -> dict[str, Any]:
        """Get comprehensive tracking statistics."""
        try:
            stats = self.tracking_stats.copy()

            # Add current state information
            stats.update(
                {
                    "active_prediction_records": len(self.prediction_records),
                    "pending_outcomes": len(self.pending_outcomes),
                    "ensemble_records": len(self.ensemble_records),
                    "performance_analyses": len(self.performance_analyses),
                    "comparison_reports": len(self.comparison_reports),
                    "cache_size": len(self.model_performance_cache)
                    + len(self.ensemble_performance_cache),
                    "is_initialized": self.is_initialized,
                },
            )

            return stats

        except Exception as e:
            self.print(failed("Failed to get tracking statistics: {e}"))
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Enhanced ML Tracker...")

            # Clear caches
            self.model_performance_cache.clear()
            self.ensemble_performance_cache.clear()

            # Close storage connections
            if hasattr(self, "storage_manager"):
                await self.storage_manager.close()

            self.logger.info("Enhanced ML Tracker cleanup completed")

        except Exception as e:
            self.print(failed("Failed to cleanup Enhanced ML Tracker: {e}"))


# Setup function for integration
async def setup_enhanced_ml_tracker(config: dict[str, Any]) -> EnhancedMLTracker | None:
    """
    Setup and return a configured Enhanced ML Tracker instance.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedMLTracker: Configured tracker instance or None if setup failed
    """
    try:
        tracker = EnhancedMLTracker(config)
        if await tracker.initialize():
            return tracker
        return None
    except Exception as e:
        system_print(failed("Failed to setup Enhanced ML Tracker: {e}"))
        return None
