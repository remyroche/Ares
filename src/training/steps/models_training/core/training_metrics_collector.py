"""
Training Metrics Collector - Comprehensive Metrics Collection System

This module provides comprehensive metrics collection throughout the training pipeline,
including pre-HPO, post-HPO, fold stability, and performance tracking.

Key Features:
- Pre-HPO baseline metrics collection
- Post-HPO performance metrics collection
- Cross-validation fold stability tracking
- Risk-Reward (RR) ratio calculation
- Markdown report generation for outcomes/
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_success


MIN_RISK_METRIC_STD = 1e-2


@dataclass
class FoldMetrics:
    """Metrics for a single cross-validation fold."""
    fold_number: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    training_time: float
    
    
@dataclass
class ModelMetrics:
    """Comprehensive metrics for a single model."""
    model_name: str
    model_type: str
    
    # Pre-HPO metrics
    pre_hpo_metrics: Dict[str, float] = field(default_factory=dict)
    pre_hpo_fold_metrics: List[FoldMetrics] = field(default_factory=list)
    pre_hpo_fold_stability: Dict[str, float] = field(default_factory=dict)
    
    # HPO metrics
    hpo_best_params: Dict[str, Any] = field(default_factory=dict)
    hpo_n_trials: int = 0
    hpo_time: float = 0.0
    
    # Post-HPO metrics
    post_hpo_metrics: Dict[str, float] = field(default_factory=dict)
    post_hpo_fold_metrics: List[FoldMetrics] = field(default_factory=list)
    post_hpo_fold_stability: Dict[str, float] = field(default_factory=dict)
    
    # Performance improvement
    metrics_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Risk-Reward metrics
    risk_reward_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None
    
    # Timing
    total_training_time: float = 0.0


@dataclass  
class TrainingSessionMetrics:
    """Comprehensive metrics for an entire training session."""
    session_id: str
    training_type: str  # analyst_base, analyst_ensemble, tactician_base, tactician_ensemble
    symbol: str
    timeframe: str
    timestamp: str
    
    # Model metrics
    model_metrics: List[ModelMetrics] = field(default_factory=list)
    
    # Overall session metrics
    total_training_time: float = 0.0
    best_model_name: str = ""
    best_model_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Data quality
    data_quality_score: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    

class TrainingMetricsCollector:
    """
    Comprehensive metrics collector for training pipeline.
    
    This class collects and aggregates metrics throughout the entire training
    pipeline, including pre-HPO, post-HPO, and fold stability metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the metrics collector."""
        self.logger = logger or system_logger.getChild("TrainingMetricsCollector")
        self.current_session: Optional[TrainingSessionMetrics] = None
        self.outcomes_dir = Path("outcomes")
        self.outcomes_dir.mkdir(exist_ok=True)
        
    def start_session(
        self, 
        training_type: str, 
        symbol: str, 
        timeframe: str
    ) -> TrainingSessionMetrics:
        """Start a new training session."""
        session_id = f"{training_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        self.current_session = TrainingSessionMetrics(
            session_id=session_id,
            training_type=training_type,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp
        )
        
        tprint_info(f"📊 Started metrics collection session: {session_id}")
        return self.current_session
    
    def collect_pre_hpo_metrics(
        self,
        model_name: str,
        model_type: str,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5
    ) -> ModelMetrics:
        """
        Collect pre-HPO baseline metrics with cross-validation.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (lightgbm, catboost, etc.)
            model: Untrained model instance
            X: Training features
            y: Training targets
            n_folds: Number of cross-validation folds
            
        Returns:
            ModelMetrics with pre-HPO metrics
        """
        try:
            tprint_info(f"📈 Collecting pre-HPO metrics for {model_name}...")
            start_time = time.time()

            # Initialize model metrics
            model_metrics = ModelMetrics(
                model_name=model_name,
                model_type=model_type
            )

            # Skip collection if model is not yet instantiated or lacks required API (e.g., TCN, neural net)
            has_fit = hasattr(model, "fit") and callable(getattr(model, "fit", None))
            has_predict = hasattr(model, "predict") and callable(getattr(model, "predict", None))
            if model is None or not (has_fit and has_predict):
                msg = (
                    f"Skipping pre-HPO metrics for {model_name} "
                    f"because the base model instance is not available or incomplete."
                )
                self.logger.info(msg)
                tprint_info(f"⏭️  {msg}")
                return model_metrics

            # Detect problem type (classification vs regression)
            is_classification = self._detect_task_type(y)
            
            # Perform cross-validation
            fold_metrics_list = []
            
            # Use KFold for regression (no stratification needed)
            # For classification with few classes, skip stratification to avoid sklearn issues
            # Always use KFold to avoid "Unknown label type: continuous" error
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            all_train_scores = []
            all_val_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                fold_start = time.time()
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                tprint_info(
                    f"🔍 Pre-HPO fold {fold}: X_train shape={X_train.shape}, X_val shape={X_val.shape}, "
                    f"features={len(X.columns)}"
                )
                
                # Train model on fold
                try:
                    # Handle different model types
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                    
                    # Get predictions
                    train_pred = model.predict(X_train)
                    val_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    train_metrics = self._calculate_metrics(y_train, train_pred, is_classification)
                    val_metrics = self._calculate_metrics(y_val, val_pred, is_classification)
                    
                    fold_time = time.time() - fold_start
                    
                    # Store fold metrics
                    fold_metrics = FoldMetrics(
                        fold_number=fold,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        training_time=fold_time
                    )
                    fold_metrics_list.append(fold_metrics)
                    
                    # Collect scores for stability calculation
                    primary_metric = self._get_primary_metric_name(is_classification)
                    all_train_scores.append(train_metrics.get(primary_metric, 0.0))
                    all_val_scores.append(val_metrics.get(primary_metric, 0.0))
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Fold {fold} failed: {e}")
                    continue
            
            # Calculate aggregated metrics
            if fold_metrics_list:
                model_metrics.pre_hpo_fold_metrics = fold_metrics_list
                model_metrics.pre_hpo_metrics = self._aggregate_fold_metrics(fold_metrics_list)
                model_metrics.pre_hpo_fold_stability = self._calculate_fold_stability(fold_metrics_list)
            
            # Calculate Risk-Reward metrics if possible
            if len(all_val_scores) > 0:
                model_metrics.risk_reward_ratio = self._calculate_risk_reward(all_val_scores)
                model_metrics.sharpe_ratio = self._calculate_sharpe_ratio(all_val_scores)
                model_metrics.sortino_ratio = self._calculate_sortino_ratio(all_val_scores)
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Pre-HPO metrics collected for {model_name} in {total_time:.2f}s")
            
            return model_metrics
            
        except Exception as e:
            self.logger.error(f"Pre-HPO metrics collection failed: {e}")
            return ModelMetrics(model_name=model_name, model_type=model_type)
    
    def collect_post_hpo_metrics(
        self,
        model_metrics: ModelMetrics,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: Dict[str, Any],
        hpo_n_trials: int,
        hpo_time: float,
        n_folds: int = 5
    ) -> ModelMetrics:
        """
        Collect post-HPO metrics after hyperparameter optimization.
        
        Args:
            model_metrics: Existing model metrics to update
            model: Optimized trained model
            X: Training features
            y: Training targets
            best_params: Best hyperparameters found
            hpo_n_trials: Number of HPO trials
            hpo_time: Time spent on HPO
            n_folds: Number of cross-validation folds
            
        Returns:
            Updated ModelMetrics with post-HPO metrics
        """
        try:
            tprint_info(f"📈 Collecting post-HPO metrics for {model_metrics.model_name}...")
            start_time = time.time()
            
            # Store HPO information
            model_metrics.hpo_best_params = best_params
            model_metrics.hpo_n_trials = hpo_n_trials
            model_metrics.hpo_time = hpo_time
            
            # Detect problem type (classification vs regression)
            is_classification = self._detect_task_type(y)
            
            # Perform cross-validation with optimized model
            fold_metrics_list = []
            
            # Ensure n_folds is at least 2 for cross-validation
            if n_folds < 2:
                self.logger.warning(f"⚠️ n_folds={n_folds} is too small, using n_folds=2")
                n_folds = 2
            
            # Always use KFold to avoid "Unknown label type: continuous" error
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            all_train_scores = []
            all_val_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                fold_start = time.time()

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                tprint_info(
                    f"🔍 Post-HPO fold {fold}: X_train shape={X_train.shape}, X_val shape={X_val.shape}, "
                    f"features={len(X.columns)}"
                )

                try:
                    # Train model with best params on fold
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                    
                    # Get predictions
                    train_pred = model.predict(X_train)
                    val_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    train_metrics = self._calculate_metrics(y_train, train_pred, is_classification)
                    val_metrics = self._calculate_metrics(y_val, val_pred, is_classification)
                    
                    fold_time = time.time() - fold_start
                    
                    # Store fold metrics
                    fold_metrics = FoldMetrics(
                        fold_number=fold,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        training_time=fold_time
                    )
                    fold_metrics_list.append(fold_metrics)
                    
                    # Collect scores for stability calculation
                    primary_metric = self._get_primary_metric_name(is_classification)
                    all_train_scores.append(train_metrics.get(primary_metric, 0.0))
                    all_val_scores.append(val_metrics.get(primary_metric, 0.0))
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Fold {fold} failed: {e}")
                    continue
            
            # Calculate aggregated metrics
            if fold_metrics_list:
                model_metrics.post_hpo_fold_metrics = fold_metrics_list
                model_metrics.post_hpo_metrics = self._aggregate_fold_metrics(fold_metrics_list)
                model_metrics.post_hpo_fold_stability = self._calculate_fold_stability(fold_metrics_list)
            
            # Calculate improvement
            model_metrics.metrics_improvement = self._calculate_improvement(
                model_metrics.pre_hpo_metrics,
                model_metrics.post_hpo_metrics
            )
            
            # Update Risk-Reward metrics
            if len(all_val_scores) > 0:
                model_metrics.risk_reward_ratio = self._calculate_risk_reward(all_val_scores)
                model_metrics.sharpe_ratio = self._calculate_sharpe_ratio(all_val_scores)
                model_metrics.sortino_ratio = self._calculate_sortino_ratio(all_val_scores)
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importance_'):
                model_metrics.feature_importance = dict(zip(X.columns, model.feature_importance_))
            elif hasattr(model, 'get_feature_importance'):
                model_metrics.feature_importance = dict(zip(X.columns, model.get_feature_importance()))
            
            total_time = time.time() - start_time
            model_metrics.total_training_time = total_time + hpo_time
            
            self.logger.info(f"✅ Post-HPO metrics collected for {model_metrics.model_name} in {total_time:.2f}s")
            
            return model_metrics
            
        except Exception as e:
            self.logger.error(f"Post-HPO metrics collection failed: {e}")
            return model_metrics
    
    def add_model_metrics(self, model_metrics: ModelMetrics):
        """Add model metrics to current session."""
        if self.current_session:
            self.current_session.model_metrics.append(model_metrics)
    
    def finalize_session(
        self,
        total_training_time: float,
        data_quality_score: float = 0.0,
        n_samples: int = 0,
        n_features: int = 0
    ) -> TrainingSessionMetrics:
        """Finalize the current training session."""
        if not self.current_session:
            raise ValueError("No active session to finalize")
        
        self.current_session.total_training_time = total_training_time
        self.current_session.data_quality_score = data_quality_score
        self.current_session.n_samples = n_samples
        self.current_session.n_features = n_features
        
        # Find best model - try r2 first (regression), then accuracy (classification)
        if self.current_session.model_metrics:
            # Try to find metric that exists in all models
            best_model = max(
                self.current_session.model_metrics,
                key=lambda m: m.post_hpo_metrics.get('r2', m.post_hpo_metrics.get('accuracy', 0.0))
            )
            self.current_session.best_model_name = best_model.model_name
            self.current_session.best_model_metrics = best_model.post_hpo_metrics
        
        tprint_success(f"✅ Training session finalized: {self.current_session.session_id}")
        return self.current_session
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        if not self.current_session:
            raise ValueError("No active session to report")
        
        session = self.current_session
        
        # Build report
        report = []
        report.append(f"# Training Report: {session.training_type}")
        report.append(f"\n**Session ID:** {session.session_id}")
        report.append(f"**Symbol:** {session.symbol}")
        report.append(f"**Timeframe:** {session.timeframe}")
        report.append(f"**Timestamp:** {session.timestamp}")
        report.append(f"**Total Training Time:** {session.total_training_time:.2f}s")
        report.append(f"\n---\n")
        
        # Data quality
        report.append(f"## Data Quality")
        report.append(f"- **Quality Score:** {session.data_quality_score:.2%}")
        report.append(f"- **Samples:** {session.n_samples:,}")
        report.append(f"- **Features:** {session.n_features:,}")
        report.append(f"\n---\n")
        
        # Best model summary
        report.append(f"## Best Model")
        report.append(f"**Name:** {session.best_model_name}")
        report.append(f"\n**Metrics:**")
        for metric, value in session.best_model_metrics.items():
            report.append(f"- {metric}: {value:.4f}")
        report.append(f"\n---\n")
        
        # Individual model details
        report.append(f"## Model Training Details\n")
        
        for model in session.model_metrics:
            report.append(f"### {model.model_name} ({model.model_type})\n")
            
            # Pre-HPO metrics
            report.append(f"#### Pre-HPO Metrics")
            for metric, value in model.pre_hpo_metrics.items():
                report.append(f"- {metric}: {value:.4f}")
            
            # Fold stability (pre-HPO)
            if model.pre_hpo_fold_stability:
                report.append(f"\n**Fold Stability (Pre-HPO):**")
                for metric, value in model.pre_hpo_fold_stability.items():
                    report.append(f"- {metric}: {value:.4f}")
            
            # HPO info
            report.append(f"\n#### Hyperparameter Optimization")
            report.append(f"- **Trials:** {model.hpo_n_trials}")
            report.append(f"- **Time:** {model.hpo_time:.2f}s")
            if model.hpo_best_params:
                report.append(f"- **Best Parameters:** {model.hpo_best_params}")
            
            # Post-HPO metrics
            report.append(f"\n#### Post-HPO Metrics")
            for metric, value in model.post_hpo_metrics.items():
                report.append(f"- {metric}: {value:.4f}")
            
            # Fold stability (post-HPO)
            if model.post_hpo_fold_stability:
                report.append(f"\n**Fold Stability (Post-HPO):**")
                for metric, value in model.post_hpo_fold_stability.items():
                    report.append(f"- {metric}: {value:.4f}")
            
            # Improvement
            if model.metrics_improvement:
                report.append(f"\n**Improvement:**")
                for metric, value in model.metrics_improvement.items():
                    report.append(f"- {metric}: {value:+.4f}")
            
            # Risk-Reward metrics
            report.append(f"\n#### Risk-Reward Metrics")
            report.append(f"- **Risk-Reward Ratio:** {model.risk_reward_ratio:.4f}")
            report.append(f"- **Sharpe Ratio:** {model.sharpe_ratio:.4f}")
            report.append(f"- **Sortino Ratio:** {model.sortino_ratio:.4f}")
            
            # Feature importance (top 10)
            if model.feature_importance:
                report.append(f"\n**Top 10 Important Features:**")
                sorted_features = sorted(
                    model.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for feature, importance in sorted_features:
                    report.append(f"- {feature}: {importance:.4f}")
            
            report.append(f"\n---\n")
        
        return "\n".join(report)
    
    def save_report(self, report: Optional[str] = None) -> Path:
        """Save the markdown report to outcomes/ directory."""
        if report is None:
            report = self.generate_markdown_report()
        
        if not self.current_session:
            raise ValueError("No active session to save")
        
        # Generate filename
        filename = f"{self.current_session.session_id}_training_report.md"
        filepath = self.outcomes_dir / filename
        
        # Write report
        with open(filepath, 'w') as f:
            f.write(report)
        
        tprint_success(f"📄 Training report saved to: {filepath}")
        return filepath
    
    def _get_primary_metric_name(self, is_classification: bool = False) -> str:
        """Get the primary metric name for model comparison."""
        return 'accuracy' if is_classification else 'r2'

    def _detect_task_type(
        self,
        y: Union[pd.Series, np.ndarray, Sequence[float], Iterable[float]]
    ) -> bool:
        """Detect whether the prediction task is classification or regression."""
        try:
            y_array = np.asarray(y)

            if y_array.size == 0:
                return False

            unique_values = np.unique(y_array)

            if unique_values.size < 2:
                return False

            is_integer_like = np.allclose(
                unique_values,
                unique_values.astype(int),
                rtol=1e-5,
                atol=1e-8,
            )
            has_few_classes = unique_values.size <= 10

            return bool(is_integer_like and has_few_classes)

        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "Task type detection failed (%s); defaulting to regression",
                exc,
            )
            return False

    def _calculate_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray, Sequence[float]],
        y_pred: Union[pd.Series, np.ndarray, Sequence[float]],
        is_classification: bool = False,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for either regression or classification."""

        metrics: Dict[str, float] = {}

        try:
            y_true_array = np.asarray(y_true)
            y_pred_array = np.asarray(y_pred)

            if is_classification:
                zero_division_default: Literal[0] = 0

                try:
                    y_pred_classes = np.round(y_pred_array).astype(int)
                    y_true_classes = y_true_array.astype(int)

                    max_class_index = max(len(np.unique(y_true_classes)) - 1, 0)
                    y_pred_classes = np.clip(y_pred_classes, 0, max_class_index)

                    metrics['accuracy'] = accuracy_score(
                        y_true_classes,
                        y_pred_classes,
                    )
                    metrics['precision'] = precision_score(
                        y_true_classes,
                        y_pred_classes,
                        zero_division=zero_division_default,
                        average='macro',
                    )
                    metrics['recall'] = recall_score(
                        y_true_classes,
                        y_pred_classes,
                        zero_division=zero_division_default,
                        average='macro',
                    )
                    metrics['f1_score'] = f1_score(
                        y_true_classes,
                        y_pred_classes,
                        zero_division=zero_division_default,
                        average='macro',
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.warning(
                        "Classification metrics calculation failed: %s",
                        exc,
                    )

            try:
                metrics['mse'] = float(mean_squared_error(y_true_array, y_pred_array))
                metrics['mae'] = float(mean_absolute_error(y_true_array, y_pred_array))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['r2'] = float(r2_score(y_true_array, y_pred_array))
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    "Regression metrics calculation failed: %s",
                    exc,
                )

            return metrics

        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Metrics calculation failed: %s", exc)
            return {}

    def _aggregate_fold_metrics(self, fold_metrics_list: List[FoldMetrics]) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        if not fold_metrics_list:
            return {}

        aggregated: Dict[str, float] = {}

        all_metric_names = set()
        for fold in fold_metrics_list:
            all_metric_names.update(fold.val_metrics.keys())

        for metric_name in all_metric_names:
            values = [fold.val_metrics.get(metric_name, 0.0) for fold in fold_metrics_list]
            aggregated[f'{metric_name}_mean'] = float(np.mean(values))
            aggregated[f'{metric_name}_std'] = float(np.std(values))

        return aggregated

    def _calculate_fold_stability(self, fold_metrics_list: List[FoldMetrics]) -> Dict[str, float]:
        """Calculate fold stability metrics."""
        if len(fold_metrics_list) < 2:
            return {}

        stability: Dict[str, float] = {}

        all_metric_names = set()
        for fold in fold_metrics_list:
            all_metric_names.update(fold.val_metrics.keys())

        for metric_name in all_metric_names:
            values = [fold.val_metrics.get(metric_name, 0.0) for fold in fold_metrics_list]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))

            cv = std_val / mean_val if mean_val != 0 else float('inf')
            stability[f'{metric_name}_cv'] = cv
            stability[f'{metric_name}_range'] = float(np.max(values) - np.min(values))

        return stability

    def _calculate_improvement(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement from pre-HPO to post-HPO."""
        improvement: Dict[str, float] = {}

        for metric_name in pre_metrics.keys():
            if metric_name.endswith('_mean') and metric_name in post_metrics:
                pre_value = pre_metrics[metric_name]
                post_value = post_metrics[metric_name]

                abs_improvement = post_value - pre_value
                rel_improvement = (abs_improvement / pre_value * 100) if pre_value != 0 else 0.0

                base_metric = metric_name.replace('_mean', '')
                improvement[f'{base_metric}_abs_improvement'] = abs_improvement
                improvement[f'{base_metric}_rel_improvement'] = rel_improvement

        return improvement

    def _calculate_risk_reward(self, scores: List[float]) -> float:
        """Calculate Risk-Reward ratio."""
        if not scores or len(scores) < 2:
            return 0.0

        mean_return = float(np.mean(scores))
        std_return = float(max(np.std(scores), MIN_RISK_METRIC_STD))

        return mean_return / std_return

    def _calculate_sharpe_ratio(self, scores: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if not scores or len(scores) < 2:
            return 0.0

        mean_return = float(np.mean(scores))
        std_return = float(max(np.std(scores), MIN_RISK_METRIC_STD))

        return (mean_return - risk_free_rate) / std_return

    def _calculate_sortino_ratio(self, scores: List[float], target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)."""
        if not scores or len(scores) < 2:
            return 0.0

        mean_return = float(np.mean(scores))
        downside_scores = [s for s in scores if s < target_return]

        downside_std = (
            float(max(np.std(downside_scores), MIN_RISK_METRIC_STD))
            if downside_scores
            else MIN_RISK_METRIC_STD
        )

        return (mean_return - target_return) / downside_std
