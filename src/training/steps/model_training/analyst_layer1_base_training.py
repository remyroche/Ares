"""
Analyst Layer 1 Base Training

Trains multiple Bagged LGBM base models with diversity defense.
Key features:
- OOF (Out-of-Fold) predictions only with burn-in period
- Incremental training for walk-forward validation  
- Diversity defense via correlation monitoring
- Excludes periods used by short_nn_sequence_template.py for training

Layer 1 Success Criteria:
- Information Coefficient (IC) > 0.03
- Probabilistic Sortino Ratio > 1.5
- Avg Expectancy (Net of Fees) > 0.1% per trade
- Pairwise Correlation < 0.75 (vs other Base models)
- Prediction Standard Dev > 0.05 (Active opinion)
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from scipy import stats

import lightgbm as lgb

try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(*args, **kwargs): print(*args)
    def tprint_success(*args, **kwargs): print(*args)
    def tprint_warning(*args, **kwargs): print(*args)
    def tprint_error(*args, **kwargs): print(*args)

from src.training.steps.model_training.analyst_multi_layer_metrics import (
    LayerMetrics, CalibrationMetrics, TradingMetrics, RiskMetrics,
    PredictiveMetrics, DiversityMetrics, StabilityMetrics, ActivityMetrics, GateMetrics,
    MultiLayerMetricsReporter,
    compute_calibration_metrics, compute_predictive_metrics,
    compute_trading_metrics, compute_risk_metrics, compute_diversity_metrics,
    generate_layer_markdown_report
)


@dataclass
class BaseModelConfig:
    """Configuration for a single base model."""
    name: str
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.02
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42


@dataclass
class BaggingConfig:
    """Configuration for bagging ensemble."""
    n_bags: int = 5
    subsample_ratio: float = 0.8
    feature_subsample_ratio: float = 0.9
    diversity_penalty: float = 0.1  # Penalty for correlated bags


DEFAULT_BASE_MODELS = [
    BaseModelConfig(name="lgbm_conservative", n_estimators=300, max_depth=4, learning_rate=0.01),
    BaseModelConfig(name="lgbm_balanced", n_estimators=500, max_depth=6, learning_rate=0.02),
    BaseModelConfig(name="lgbm_aggressive", n_estimators=700, max_depth=8, learning_rate=0.03),
]


class DiversityDefense:
    """
    Diversity defense mechanism for bagged models.
    
    Monitors and enforces diversity across bags/models to prevent
    overfitting and ensure robust ensemble predictions.
    """
    
    def __init__(
        self,
        max_pairwise_correlation: float = 0.75,
        min_prediction_std: float = 0.05
    ):
        """
        Initialize diversity defense.
        
        Args:
            max_pairwise_correlation: Maximum allowed correlation between models
            min_prediction_std: Minimum required prediction standard deviation
        """
        self.max_pairwise_correlation = max_pairwise_correlation
        self.min_prediction_std = min_prediction_std
        self.correlation_history: List[float] = []
    
    def check_diversity(
        self,
        predictions_matrix: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if predictions meet diversity requirements.
        
        Args:
            predictions_matrix: Matrix of shape (n_samples, n_models)
            
        Returns:
            (passed, metrics_dict)
        """
        predictions_matrix = np.asarray(predictions_matrix)
        
        if predictions_matrix.ndim != 2 or predictions_matrix.shape[1] < 2:
            return True, {"pairwise_correlation": 0.0, "prediction_std": 0.0}
        
        # Compute pairwise correlations
        n_models = predictions_matrix.shape[1]
        correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr, _ = stats.pearsonr(
                    predictions_matrix[:, i],
                    predictions_matrix[:, j]
                )
                if np.isfinite(corr):
                    correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        max_correlation = np.max(correlations) if correlations else 0.0
        
        # Compute average prediction std across models
        pred_stds = np.std(predictions_matrix, axis=0)
        avg_pred_std = np.mean(pred_stds)
        
        self.correlation_history.append(avg_correlation)
        
        metrics = {
            "pairwise_correlation": avg_correlation,
            "max_pairwise_correlation": max_correlation,
            "prediction_std": avg_pred_std,
            "n_models": n_models
        }
        
        # Check diversity criteria
        passed = (
            avg_correlation < self.max_pairwise_correlation and
            avg_pred_std > self.min_prediction_std
        )
        
        return passed, metrics
    
    def apply_diversity_penalty(
        self,
        predictions_matrix: np.ndarray,
        penalty_weight: float = 0.1
    ) -> np.ndarray:
        """
        Apply diversity-aware weighting to predictions.
        
        Args:
            predictions_matrix: Matrix of shape (n_samples, n_models)
            penalty_weight: Weight for diversity penalty
            
        Returns:
            Diversity-adjusted weights for each model
        """
        predictions_matrix = np.asarray(predictions_matrix)
        n_models = predictions_matrix.shape[1]
        
        if n_models < 2:
            return np.ones(n_models) / n_models
        
        # Compute uniqueness score for each model
        uniqueness_scores = np.zeros(n_models)
        
        for i in range(n_models):
            correlations = []
            for j in range(n_models):
                if i != j:
                    corr, _ = stats.pearsonr(
                        predictions_matrix[:, i],
                        predictions_matrix[:, j]
                    )
                    if np.isfinite(corr):
                        correlations.append(abs(corr))
            
            # Lower average correlation = higher uniqueness
            avg_corr = np.mean(correlations) if correlations else 0.0
            uniqueness_scores[i] = 1 - avg_corr
        
        # Convert to weights
        weights = uniqueness_scores * penalty_weight + (1 - penalty_weight)
        weights = weights / weights.sum()
        
        return weights


class BaggedLGBMTrainer:
    """
    Trains Bagged LGBM models with diversity defense.
    
    Key features:
    - Multiple bags with bootstrap sampling
    - Feature subsampling for diversity
    - OOF predictions with proper walk-forward validation
    - Burn-in period handling
    """
    
    def __init__(
        self,
        base_config: BaseModelConfig,
        bagging_config: BaggingConfig,
        diversity_defense: DiversityDefense,
        burn_in_periods: int = 100,
        nn_sequence_lookback: int = 24  # Bars used by short_nn_sequence_template
    ):
        """
        Initialize the trainer.
        
        Args:
            base_config: Configuration for the base LGBM model
            bagging_config: Configuration for bagging
            diversity_defense: Diversity defense mechanism
            burn_in_periods: Number of periods to exclude as burn-in
            nn_sequence_lookback: Lookback used by NN sequence encoder (to exclude from training)
        """
        self.base_config = base_config
        self.bagging_config = bagging_config
        self.diversity_defense = diversity_defense
        self.burn_in_periods = burn_in_periods
        self.nn_sequence_lookback = nn_sequence_lookback
        
        self.bag_models: List[lgb.LGBMClassifier] = []
        self.bag_feature_indices: List[np.ndarray] = []
        self.feature_names: List[str] = []
        self.oof_predictions: Optional[pd.Series] = None
        self.training_metrics: Dict[str, Any] = {}
    
    def _create_base_model(self, random_state: int = 42) -> lgb.LGBMClassifier:
        """Create a base LGBM model with configured parameters."""
        return lgb.LGBMClassifier(
            n_estimators=self.base_config.n_estimators,
            max_depth=self.base_config.max_depth,
            learning_rate=self.base_config.learning_rate,
            num_leaves=self.base_config.num_leaves,
            min_child_samples=self.base_config.min_child_samples,
            subsample=self.base_config.subsample,
            colsample_bytree=self.base_config.colsample_bytree,
            reg_alpha=self.base_config.reg_alpha,
            reg_lambda=self.base_config.reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            importance_type='gain'
        )
    
    def _get_valid_training_mask(
        self,
        n_samples: int,
        nn_embedding_start_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get mask for valid training samples, excluding:
        - Burn-in period
        - Samples used by NN sequence encoder for training
        
        Args:
            n_samples: Total number of samples
            nn_embedding_start_idx: Index where NN embeddings become valid
            
        Returns:
            Boolean mask for valid training samples
        """
        mask = np.ones(n_samples, dtype=bool)
        
        # Exclude burn-in period
        mask[:self.burn_in_periods] = False
        
        # Exclude NN sequence lookback period if applicable
        if nn_embedding_start_idx is not None:
            # NN embeddings are only valid after lookback period
            # We should NOT train on samples where NN features are computed from training data
            mask[:nn_embedding_start_idx + self.nn_sequence_lookback] = False
        
        return mask
    
    def train_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        val_ratio: float = 0.2,
        embargo_periods: int = 10,
        nn_embedding_start_idx: Optional[int] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Train using walk-forward validation with OOF predictions.
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_splits: Number of walk-forward splits
            val_ratio: Validation ratio per split
            embargo_periods: Embargo periods between train and val
            nn_embedding_start_idx: Index where NN embeddings become valid
            
        Returns:
            (OOF predictions Series, training metrics dict)
        """
        start_time = time.time()
        
        self.feature_names = list(X.columns)
        n_samples = len(X)
        
        # Get valid training mask
        valid_mask = self._get_valid_training_mask(n_samples, nn_embedding_start_idx)
        valid_indices = np.where(valid_mask)[0]
        
        tprint_info(f"[{self.base_config.name}] Valid samples: {len(valid_indices)}/{n_samples}")
        
        # Initialize OOF predictions
        oof_predictions = np.full(n_samples, np.nan)
        fold_metrics: List[Dict[str, float]] = []
        all_bag_predictions: List[np.ndarray] = []
        
        # Walk-forward splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(valid_indices)):
            # Map back to original indices
            train_idx = valid_indices[train_idx]
            val_idx = valid_indices[val_idx]
            
            # Apply embargo
            if embargo_periods > 0 and len(train_idx) > embargo_periods:
                train_idx = train_idx[:-embargo_periods]
            
            tprint_info(f"[{self.base_config.name}] Fold {fold_idx + 1}/{n_splits}: "
                       f"Train={len(train_idx)}, Val={len(val_idx)}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Train bags for this fold
            fold_bag_predictions = []
            
            for bag_idx in range(self.bagging_config.n_bags):
                # Bootstrap sample
                rng = np.random.RandomState(self.base_config.random_state + fold_idx * 100 + bag_idx)
                boot_idx = rng.choice(
                    len(X_train),
                    size=int(len(X_train) * self.bagging_config.subsample_ratio),
                    replace=True
                )
                
                # Feature subsample
                n_features = X_train.shape[1]
                n_feat_sample = int(n_features * self.bagging_config.feature_subsample_ratio)
                feat_idx = rng.choice(n_features, size=n_feat_sample, replace=False)
                
                X_boot = X_train.iloc[boot_idx, feat_idx]
                y_boot = y_train.iloc[boot_idx]
                
                # Train model
                model = self._create_base_model(
                    random_state=self.base_config.random_state + bag_idx
                )
                
                model.fit(
                    X_boot, y_boot,
                    eval_set=[(X_val.iloc[:, feat_idx], y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                # Predict on validation set
                val_probs = model.predict_proba(X_val.iloc[:, feat_idx])[:, 1]
                fold_bag_predictions.append(val_probs)
                
                # Store model info (only from last fold for inference)
                if fold_idx == n_splits - 1:
                    self.bag_models.append(model)
                    self.bag_feature_indices.append(feat_idx)
            
            # Aggregate bag predictions with diversity weighting
            fold_bag_matrix = np.column_stack(fold_bag_predictions)
            
            # Apply diversity defense
            diversity_passed, diversity_metrics = self.diversity_defense.check_diversity(fold_bag_matrix)
            if not diversity_passed:
                tprint_warning(f"[{self.base_config.name}] Fold {fold_idx + 1}: "
                              f"Diversity check failed: {diversity_metrics}")
            
            # Get diversity-adjusted weights
            weights = self.diversity_defense.apply_diversity_penalty(
                fold_bag_matrix,
                penalty_weight=self.bagging_config.diversity_penalty
            )
            
            # Weighted average of bag predictions
            fold_predictions = np.average(fold_bag_matrix, axis=1, weights=weights)
            oof_predictions[val_idx] = fold_predictions
            
            all_bag_predictions.append(fold_bag_matrix)
            
            # Compute fold metrics
            if len(np.unique(y_val)) > 1:
                from sklearn.metrics import roc_auc_score, log_loss
                fold_auc = roc_auc_score(y_val, fold_predictions)
                fold_logloss = log_loss(y_val, fold_predictions)
            else:
                fold_auc = 0.5
                fold_logloss = 0.0
            
            fold_metrics.append({
                "fold": fold_idx + 1,
                "auc": fold_auc,
                "logloss": fold_logloss,
                "diversity": diversity_metrics,
                "n_train": len(train_idx),
                "n_val": len(val_idx)
            })
        
        training_time = time.time() - start_time
        
        # Create OOF Series
        self.oof_predictions = pd.Series(oof_predictions, index=X.index, name=f"{self.base_config.name}_prob")
        
        # Aggregate training metrics
        self.training_metrics = {
            "model_name": self.base_config.name,
            "n_splits": n_splits,
            "n_bags": self.bagging_config.n_bags,
            "mean_auc": np.mean([m["auc"] for m in fold_metrics]),
            "std_auc": np.std([m["auc"] for m in fold_metrics]),
            "mean_logloss": np.mean([m["logloss"] for m in fold_metrics]),
            "fold_metrics": fold_metrics,
            "training_time_sec": training_time,
            "n_features": len(self.feature_names),
            "n_valid_samples": len(valid_indices),
            "n_total_samples": n_samples
        }
        
        tprint_success(f"[{self.base_config.name}] Training complete: "
                      f"AUC={self.training_metrics['mean_auc']:.4f} ± {self.training_metrics['std_auc']:.4f}")
        
        return self.oof_predictions, self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using trained bag ensemble.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Probability predictions
        """
        if not self.bag_models:
            raise ValueError("Model not trained yet")
        
        bag_predictions = []
        
        for model, feat_idx in zip(self.bag_models, self.bag_feature_indices):
            X_subset = X.iloc[:, feat_idx]
            probs = model.predict_proba(X_subset)[:, 1]
            bag_predictions.append(probs)
        
        bag_matrix = np.column_stack(bag_predictions)
        weights = self.diversity_defense.apply_diversity_penalty(
            bag_matrix,
            penalty_weight=self.bagging_config.diversity_penalty
        )
        
        return np.average(bag_matrix, axis=1, weights=weights)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance across bags."""
        if not self.bag_models:
            return pd.DataFrame()
        
        importance_dict = {}
        
        for model, feat_idx in zip(self.bag_models, self.bag_feature_indices):
            importances = model.feature_importances_
            feature_names = [self.feature_names[i] for i in feat_idx]
            
            for name, imp in zip(feature_names, importances):
                if name not in importance_dict:
                    importance_dict[name] = []
                importance_dict[name].append(imp)
        
        # Average importances
        avg_importance = {
            name: np.mean(imps) for name, imps in importance_dict.items()
        }
        
        df = pd.DataFrame({
            "feature": list(avg_importance.keys()),
            "importance": list(avg_importance.values())
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


class Layer1Orchestrator:
    """
    Orchestrates training of multiple Layer 1 base models.
    
    Manages:
    - Multiple base model configurations
    - Diversity monitoring across models
    - OOF prediction aggregation
    - Metrics reporting
    """
    
    def __init__(
        self,
        base_model_configs: Optional[List[BaseModelConfig]] = None,
        bagging_config: Optional[BaggingConfig] = None,
        reporter: Optional[MultiLayerMetricsReporter] = None,
        symbol: str = "UNKNOWN",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            base_model_configs: List of base model configurations
            bagging_config: Bagging configuration
            reporter: Metrics reporter instance
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
        """
        self.base_model_configs = base_model_configs or DEFAULT_BASE_MODELS
        self.bagging_config = bagging_config or BaggingConfig()
        self.reporter = reporter or MultiLayerMetricsReporter()
        
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.direction = direction
        
        self.trainers: Dict[str, BaggedLGBMTrainer] = {}
        self.oof_predictions: Dict[str, pd.Series] = {}
        self.all_metrics: List[LayerMetrics] = []
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None,
        n_splits: int = 5,
        nn_embedding_start_idx: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Train all base models and collect OOF predictions.
        
        Args:
            X: Features DataFrame
            y: Target Series (binary labels)
            returns: Optional returns for trading metrics
            n_splits: Number of walk-forward splits
            nn_embedding_start_idx: Index where NN embeddings become valid
            
        Returns:
            Dict of model_name -> OOF predictions Series
        """
        tprint_info("=" * 80)
        tprint_info("LAYER 1: TRAINING BASE MODELS")
        tprint_info("=" * 80)
        
        for config in self.base_model_configs:
            tprint_info(f"\n📊 Training: {config.name}")
            tprint_info("-" * 40)
            
            # Create trainer
            diversity_defense = DiversityDefense(
                max_pairwise_correlation=0.75,
                min_prediction_std=0.05
            )
            
            trainer = BaggedLGBMTrainer(
                base_config=config,
                bagging_config=self.bagging_config,
                diversity_defense=diversity_defense,
                burn_in_periods=100,
                nn_sequence_lookback=24
            )
            
            # Train and get OOF predictions
            oof_preds, training_metrics = trainer.train_walk_forward(
                X, y,
                n_splits=n_splits,
                nn_embedding_start_idx=nn_embedding_start_idx
            )
            
            self.trainers[config.name] = trainer
            self.oof_predictions[config.name] = oof_preds
            
            # Compute comprehensive metrics
            layer_metrics = self._compute_layer_metrics(
                config.name,
                oof_preds,
                y,
                returns,
                training_metrics
            )
            
            self.all_metrics.append(layer_metrics)
            self.reporter.record_metrics(layer_metrics)
            
            # Generate individual markdown report
            report_path = self.reporter.output_dir / f"L1_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            generate_layer_markdown_report(layer_metrics, str(report_path))
        
        # Check cross-model diversity
        self._check_cross_model_diversity()
        
        return self.oof_predictions
    
    def _compute_layer_metrics(
        self,
        model_name: str,
        oof_preds: pd.Series,
        y: pd.Series,
        returns: Optional[pd.Series],
        training_metrics: Dict[str, Any]
    ) -> LayerMetrics:
        """Compute comprehensive Layer 1 metrics."""
        # Align and filter NaN
        valid_mask = ~oof_preds.isna()
        y_valid = y[valid_mask].values
        preds_valid = oof_preds[valid_mask].values
        
        if returns is not None:
            returns_valid = returns[valid_mask].values
        else:
            returns_valid = None
        
        # Compute sub-metrics
        calibration = compute_calibration_metrics(y_valid, preds_valid)
        predictive = compute_predictive_metrics(y_valid, preds_valid, returns_valid)
        
        if returns_valid is not None:
            trading = compute_trading_metrics(preds_valid, returns_valid, threshold=0.5)
            risk = compute_risk_metrics(returns_valid)
        else:
            trading = TradingMetrics()
            risk = RiskMetrics()
        
        # Create LayerMetrics
        metrics = LayerMetrics(
            model_name=model_name,
            layer="L1_base",
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            direction=self.direction,
            model_type="bagged_lgbm",
            calibration=calibration,
            stability=StabilityMetrics(),
            trading=trading,
            risk=risk,
            predictive=predictive,
            activity=ActivityMetrics(),
            diversity=DiversityMetrics(),  # Will be filled by cross-model check
            gate=GateMetrics(),
            n_samples=int(valid_mask.sum()),
            n_features=training_metrics.get("n_features", 0),
            training_duration_sec=training_metrics.get("training_time_sec", 0),
            notes=f"AUC: {training_metrics.get('mean_auc', 0):.4f}"
        )
        
        return metrics
    
    def _check_cross_model_diversity(self) -> None:
        """Check diversity across all trained models."""
        if len(self.oof_predictions) < 2:
            return
        
        # Stack all OOF predictions
        preds_df = pd.DataFrame(self.oof_predictions)
        preds_df = preds_df.dropna()
        
        if preds_df.empty:
            tprint_warning("⚠️ Cannot check cross-model diversity: no valid overlapping predictions")
            return
        
        diversity = compute_diversity_metrics(preds_df.values)
        
        tprint_info("\n📊 Cross-Model Diversity Metrics:")
        tprint_info(f"   Avg Pairwise Correlation: {diversity.pairwise_correlation:.4f}")
        tprint_info(f"   Max Pairwise Correlation: {diversity.max_pairwise_correlation:.4f}")
        tprint_info(f"   Min Pairwise Correlation: {diversity.min_pairwise_correlation:.4f}")
        
        # Check threshold
        if diversity.pairwise_correlation < 0.75:
            tprint_success("✅ Diversity check PASSED: Models are sufficiently diverse")
        else:
            tprint_warning("⚠️ Diversity check FAILED: Models are too correlated")
        
        # Update metrics with diversity info
        for metrics in self.all_metrics:
            metrics.diversity = diversity
    
    def get_combined_oof_predictions(self) -> pd.DataFrame:
        """Get all OOF predictions as a DataFrame."""
        return pd.DataFrame(self.oof_predictions)
    
    def get_base_model_trainers(self) -> Dict[str, BaggedLGBMTrainer]:
        """Get all trained model trainers."""
        return self.trainers


# End of module
