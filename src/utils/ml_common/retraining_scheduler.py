"""
Retraining Scheduler for ML Models.

Manages retraining schedules for different model types (HMM, GMM, XGB)
and generates out-of-fold (OOF) predictions to prevent lookahead bias.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrainingSchedule:
    """Configuration for model retraining schedule."""

    model_type: str  # 'hmm', 'gmm', 'xgb', 'analyst_base', 'analyst_ensemble'
    retrain_interval_days: int  # How often to retrain
    burnin_pct: float  # Burn-in period as percentage of data
    min_samples_for_training: int = 1000  # Minimum samples needed
    enable_warm_start: bool = True  # Whether to use warm start

    @classmethod
    def for_hmm(cls) -> 'RetrainingSchedule':
        """Standard schedule for HMM models."""
        return cls(
            model_type='hmm',
            retrain_interval_days=15,
            burnin_pct=1/12,  # 3 months
            min_samples_for_training=1000,
            enable_warm_start=True
        )

    @classmethod
    def for_gmm(cls) -> 'RetrainingSchedule':
        """Standard schedule for GMM models."""
        return cls(
            model_type='gmm',
            retrain_interval_days=15,
            burnin_pct=1/12,  # 3 months
            min_samples_for_training=1000,
            enable_warm_start=True
        )

    @classmethod
    def for_xgb(cls) -> 'RetrainingSchedule':
        """Standard schedule for XGB models."""
        return cls(
            model_type='xgb',
            retrain_interval_days=5,
            burnin_pct=1/12,  # 3 months
            min_samples_for_training=1000,
            enable_warm_start=False  # XGB doesn't use warm start
        )

    @classmethod
    def for_analyst_base(cls, execution_mode: str = 'blank') -> 'RetrainingSchedule':
        """
        Standard schedule for analyst base models.
        
        Burn-in = full_period - 4 months:
        - blank mode: 360 days - 120 days = 240 days (~8 months burn-in)
        - full mode: 1095 days - 120 days = 975 days (~32 months burn-in)
        - light mode: 30 days - 120 days = 0 (no burn-in, uses minimum)
        
        Args:
            execution_mode: One of 'light', 'blank', 'full'
        """
        # Mode lookback days (from ares_launcher)
        MODE_LOOKBACK_DAYS = {
            "light": 30,
            "blank": 360,  # 1 year
            "full": 365 * 3  # 3 years
        }
        
        full_period_days = MODE_LOOKBACK_DAYS.get(execution_mode, 360)
        burn_in_buffer_days = 120  # 4 months
        
        # Calculate burn-in as full_period - 4 months
        burn_in_days = max(30, full_period_days - burn_in_buffer_days)
        
        # Convert to percentage for backward compatibility
        burnin_pct = burn_in_days / full_period_days if full_period_days > 0 else 0.5
        
        return cls(
            model_type='analyst_base',
            retrain_interval_days=14,  # 14-day OOF batches for incremental training
            burnin_pct=burnin_pct,
            min_samples_for_training=500,  # Lowered for incremental approach
            enable_warm_start=True  # Enable warm start for incremental training
        )

    @classmethod
    def for_analyst_ensemble(cls) -> 'RetrainingSchedule':
        """Standard schedule for analyst ensemble models."""
        return cls(
            model_type='analyst_ensemble',
            retrain_interval_days=5,  # Can train more frequently as it uses OOF predictions
            burnin_pct=0.0,  # No burn-in, uses predictions from base models
            min_samples_for_training=2000,
            enable_warm_start=False
        )


@dataclass
class TrainingWindow:
    """Defines a training window for OOF prediction generation."""

    training_start: datetime
    training_end: datetime
    prediction_start: datetime
    prediction_end: datetime
    window_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'training_start': self.training_start.isoformat(),
            'training_end': self.training_end.isoformat(),
            'prediction_start': self.prediction_start.isoformat(),
            'prediction_end': self.prediction_end.isoformat(),
            'window_id': self.window_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingWindow':
        """Create from dictionary."""
        return cls(
            training_start=datetime.fromisoformat(data['training_start']),
            training_end=datetime.fromisoformat(data['training_end']),
            prediction_start=datetime.fromisoformat(data['prediction_start']),
            prediction_end=datetime.fromisoformat(data['prediction_end']),
            window_id=data['window_id']
        )


class OOFPredictionGenerator:
    """
    Generates out-of-fold predictions to prevent lookahead bias.

    Models are only trained on data up to time t to make predictions at time t,
    ensuring no future information leaks into the model.
    """

    def __init__(
        self,
        schedule: RetrainingSchedule,
        data_start: datetime,
        data_end: datetime
    ):
        """
        Initialize OOF prediction generator.

        Args:
            schedule: Retraining schedule configuration
            data_start: Start of available data
            data_end: End of available data
        """
        self.schedule = schedule
        self.data_start = data_start
        self.data_end = data_end
        self.windows = self._create_training_windows()

    def _create_training_windows(self) -> List[TrainingWindow]:
        """
        Create training windows based on retraining schedule.

        Returns:
            List of TrainingWindow objects defining when to train and predict
        """
        windows = []

        # Calculate burn-in period
        total_duration = (self.data_end - self.data_start).days
        burnin_days = int(total_duration * self.schedule.burnin_pct)
        burnin_end = self.data_start + timedelta(days=burnin_days)

        # Start making predictions after burn-in
        current_prediction_start = burnin_end
        window_id = 0

        while current_prediction_start < self.data_end:
            # Training period: from data start to current prediction start
            training_start = self.data_start
            training_end = current_prediction_start

            # Prediction period: next retrain_interval_days
            prediction_end = min(
                current_prediction_start + timedelta(days=self.schedule.retrain_interval_days),
                self.data_end
            )

            window = TrainingWindow(
                training_start=training_start,
                training_end=training_end,
                prediction_start=current_prediction_start,
                prediction_end=prediction_end,
                window_id=window_id
            )
            windows.append(window)

            # Move to next window
            current_prediction_start = prediction_end
            window_id += 1

        return windows

    def generate_oof_predictions(
        self,
        data: pd.DataFrame,
        training_func: Callable[[pd.DataFrame], Any],
        prediction_func: Callable[[Any, pd.DataFrame], pd.DataFrame],
        show_progress: bool = True
    ) -> Tuple[pd.DataFrame, List[Any], List[Dict[str, Any]]]:
        """
        Generate out-of-fold predictions using retraining windows.

        Args:
            data: Full dataset with DatetimeIndex
            training_func: Function that takes training data and returns a trained model
            prediction_func: Function that takes (model, data) and returns predictions DataFrame
            show_progress: Whether to show progress messages

        Returns:
            Tuple of (oof_predictions_df, models_list, metadata_list)
            - oof_predictions_df: DataFrame with predictions for all timestamps
            - models_list: List of trained models for each window
            - metadata_list: List of metadata dicts for each window
        """
        all_predictions = []
        all_models = []
        all_metadata = []

        for window in self.windows:
            if show_progress:
                logger.info(
                    f"Training window {window.window_id}: "
                    f"Train on {window.training_start} to {window.training_end}, "
                    f"Predict {window.prediction_start} to {window.prediction_end}"
                )

            # Get training data (only data before prediction period)
            train_mask = (data.index >= window.training_start) & (data.index < window.training_end)
            train_data = data.loc[train_mask]

            # Check minimum samples
            if len(train_data) < self.schedule.min_samples_for_training:
                logger.warning(
                    f"Window {window.window_id}: Insufficient training samples "
                    f"({len(train_data)} < {self.schedule.min_samples_for_training}). Skipping."
                )
                continue

            # Train model on data up to this point
            model = training_func(train_data)
            all_models.append(model)

            # Get prediction data
            pred_mask = (data.index >= window.prediction_start) & (data.index <= window.prediction_end)
            pred_data = data.loc[pred_mask]

            if len(pred_data) == 0:
                continue

            # Make predictions
            predictions = prediction_func(model, pred_data)
            all_predictions.append(predictions)

            # Store metadata
            metadata = {
                'window_id': window.window_id,
                'training_samples': len(train_data),
                'prediction_samples': len(pred_data),
                'training_start': window.training_start.isoformat(),
                'training_end': window.training_end.isoformat(),
                'prediction_start': window.prediction_start.isoformat(),
                'prediction_end': window.prediction_end.isoformat()
            }
            all_metadata.append(metadata)

        # Combine all predictions
        if all_predictions:
            oof_predictions = pd.concat(all_predictions, axis=0)
            # Sort by index to ensure chronological order
            oof_predictions = oof_predictions.sort_index()
        else:
            oof_predictions = pd.DataFrame()

        return oof_predictions, all_models, all_metadata

    def save_metadata(self, filepath: Path, metadata_list: List[Dict[str, Any]]):
        """Save training metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'schedule': {
                    'model_type': self.schedule.model_type,
                    'retrain_interval_days': self.schedule.retrain_interval_days,
                    'burnin_pct': self.schedule.burnin_pct
                },
                'data_start': self.data_start.isoformat(),
                'data_end': self.data_end.isoformat(),
                'windows': metadata_list
            }, f, indent=2)


class RetrainingManager:
    """
    Manages model retraining based on last training timestamp.

    Determines if a model needs retraining based on:
    - Time since last training
    - Retraining schedule
    - Availability of new data
    """

    def __init__(self, cache_dir: Path = Path("cache/model_retraining")):
        """
        Initialize retraining manager.

        Args:
            cache_dir: Directory to store retraining metadata
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_metadata_path(self, model_id: str) -> Path:
        """Get path to retraining metadata file."""
        return self.cache_dir / f"{model_id}_retraining.json"

    def get_last_training_time(self, model_id: str) -> Optional[datetime]:
        """
        Get the last time a model was trained.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Last training timestamp, or None if never trained
        """
        metadata_path = self._get_metadata_path(model_id)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return datetime.fromisoformat(metadata['last_training_time'])
        except (json.JSONDecodeError, KeyError):
            return None

    def should_retrain(
        self,
        model_id: str,
        schedule: RetrainingSchedule,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Determine if a model should be retrained.

        Args:
            model_id: Unique identifier for the model
            schedule: Retraining schedule configuration
            current_time: Current time (default: now)

        Returns:
            True if model should be retrained
        """
        if current_time is None:
            current_time = datetime.now()

        last_training = self.get_last_training_time(model_id)

        # If never trained, should retrain
        if last_training is None:
            return True

        # Check if enough time has passed
        days_since_training = (current_time - last_training).days
        return days_since_training >= schedule.retrain_interval_days

    def record_training(
        self,
        model_id: str,
        schedule: RetrainingSchedule,
        training_time: Optional[datetime] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record that a model was trained.

        Args:
            model_id: Unique identifier for the model
            schedule: Retraining schedule configuration
            training_time: Time of training (default: now)
            additional_metadata: Additional metadata to store
        """
        if training_time is None:
            training_time = datetime.now()

        metadata = {
            'model_id': model_id,
            'last_training_time': training_time.isoformat(),
            'model_type': schedule.model_type,
            'retrain_interval_days': schedule.retrain_interval_days
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        metadata_path = self._get_metadata_path(model_id)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Recorded training for model {model_id} at {training_time}")


def create_sample_weights(
    timestamps: pd.DatetimeIndex,
    half_life_months: float = 18.0
) -> np.ndarray:
    """
    Create exponential sample weights giving more importance to recent samples.

    Uses exponential decay: weight = exp(-decay_rate * months_ago)
    where decay_rate = ln(2) / half_life_months

    Args:
        timestamps: DatetimeIndex of samples
        half_life_months: Half-life period in months (default: 18 months)

    Returns:
        Array of sample weights normalized to sum to 1
    """
    # Get most recent timestamp
    max_time = timestamps.max()

    # Calculate months ago for each sample
    # Ensure we operate on NumPy arrays (not Index objects) for numeric ops
    deltas = max_time - timestamps
    days_ago = np.asarray(deltas.days, dtype=float)
    months_ago = days_ago / 30.44  # Average days per month

    # Calculate decay rate
    decay_rate = np.log(2) / half_life_months

    # Calculate weights using exponential decay
    weights = np.exp(-decay_rate * months_ago)

    # Normalize to sum to 1
    total = float(weights.sum())
    if total > 0:
        weights = weights / total

    return weights
