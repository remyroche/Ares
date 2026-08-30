"""
Analyst Layer 2 Meta Training

Trains meta models on base model OOF predictions + disagreement features.
Supports multiple modalities:
- Simple Average (baseline)
- Well-regularized LGBM
- Extra Trees
- Linear Model (Ridge/Logistic Regression)

Layer 2 Success Criteria:
- IC Improvement: +20% vs Best Base Model
- Log Loss Reduction: Lower than Best Base Model
- Better PnL and Sortino, Lower Max Drawdown vs Average
- Expected Calibration Error (ECE): < 0.05
- Brier Score: Lower than Simple Average
- Feature Stability: Top 3 features consistent across folds
- Total Profit Factor: > 1.8
- PnL: Superior to Best Base Model
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
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
    compute_trading_metrics, compute_risk_metrics,
    generate_layer_markdown_report
)

from src.feature_generation.categories.ensemble_disagreement import (
    EnsembleDisagreementFeatures,
    calculate_ensemble_disagreement_features,
    get_core_feature_names
)


class MetaModelType(Enum):
    """Meta model types."""
    SIMPLE_AVERAGE = "simple_average"
    REGULARIZED_LGBM = "regularized_lgbm"
    EXTRA_TREES = "extra_trees"
    LINEAR = "linear"


@dataclass
class MetaModelConfig:
    """Configuration for meta model."""
    model_type: MetaModelType
    name: str
    
    # LGBM specific
    lgbm_n_estimators: int = 200
    lgbm_max_depth: int = 4
    lgbm_learning_rate: float = 0.01
    lgbm_reg_alpha: float = 1.0
    lgbm_reg_lambda: float = 1.0
    lgbm_min_child_samples: int = 100
    
    # ExtraTrees specific
    et_n_estimators: int = 300
    et_max_depth: int = 6
    et_min_samples_leaf: int = 50
    et_max_features: str = "sqrt"
    
    # Linear specific
    linear_C: float = 0.1
    linear_solver: str = "lbfgs"
    
    # Common
    random_state: int = 42


DEFAULT_META_CONFIGS = [
    MetaModelConfig(
        model_type=MetaModelType.SIMPLE_AVERAGE,
        name="meta_simple_average"
    ),
    MetaModelConfig(
        model_type=MetaModelType.REGULARIZED_LGBM,
        name="meta_lgbm",
        lgbm_n_estimators=200,
        lgbm_max_depth=4,
        lgbm_reg_alpha=1.0,
        lgbm_reg_lambda=1.0
    ),
    MetaModelConfig(
        model_type=MetaModelType.EXTRA_TREES,
        name="meta_extratrees",
        et_n_estimators=300,
        et_max_depth=6,
        et_min_samples_leaf=50
    ),
    MetaModelConfig(
        model_type=MetaModelType.LINEAR,
        name="meta_linear",
        linear_C=0.1
    )
]


class DisagreementFeatureGenerator:
    """
    Generates disagreement features from base model predictions.
    
    Uses the centralized EnsembleDisagreementFeatures class.
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.calculator = EnsembleDisagreementFeatures()
    
    def generate(
        self,
        base_predictions: pd.DataFrame,
        prefix: str = "disagree_"
    ) -> pd.DataFrame:
        """
        Generate disagreement features from base model predictions.
        
        Args:
            base_predictions: DataFrame with base model predictions (columns = models)
            prefix: Prefix for feature names
            
        Returns:
            DataFrame with disagreement features
        """
        if base_predictions.empty or base_predictions.shape[1] < 2:
            tprint_warning("⚠️ Not enough base models for disagreement features")
            return pd.DataFrame(index=base_predictions.index)
        
        # Prepare inputs for the disagreement calculator
        model_predictions = {}
        model_probabilities = {}
        
        for col in base_predictions.columns:
            preds = base_predictions[col].values
            model_predictions[col] = preds
            # Convert to probability format (already probabilities)
            model_probabilities[col] = preds
        
        # Calculate disagreement features
        features = calculate_ensemble_disagreement_features(
            model_predictions=model_predictions,
            model_probabilities=model_probabilities
        )
        
        # Convert to DataFrame
        disagree_df = pd.DataFrame(index=base_predictions.index)
        
        for feat_name, feat_series in features.items():
            disagree_df[f"{prefix}{feat_name}"] = feat_series.reindex(base_predictions.index).values
        
        # Add additional custom features
        disagree_df[f"{prefix}mean"] = base_predictions.mean(axis=1)
        disagree_df[f"{prefix}std"] = base_predictions.std(axis=1)
        disagree_df[f"{prefix}min"] = base_predictions.min(axis=1)
        disagree_df[f"{prefix}max"] = base_predictions.max(axis=1)
        disagree_df[f"{prefix}median"] = base_predictions.median(axis=1)
        disagree_df[f"{prefix}skew"] = base_predictions.apply(lambda x: stats.skew(x), axis=1)
        
        return disagree_df


class MetaModelTrainer:
    """
    Trains a single meta model on base predictions + disagreement features.
    """
    
    def __init__(
        self,
        config: MetaModelConfig,
        burn_in_periods: int = 100
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Meta model configuration
            burn_in_periods: Number of periods to exclude as burn-in
        """
        self.config = config
        self.burn_in_periods = burn_in_periods
        
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self.oof_predictions: Optional[pd.Series] = None
        self.training_metrics: Dict[str, Any] = {}
        self.fold_feature_importances: List[Dict[str, float]] = []
    
    def _create_model(self):
        """Create the meta model based on config."""
        if self.config.model_type == MetaModelType.SIMPLE_AVERAGE:
            return None  # Simple average doesn't need a model
        
        elif self.config.model_type == MetaModelType.REGULARIZED_LGBM:
            return lgb.LGBMClassifier(
                n_estimators=self.config.lgbm_n_estimators,
                max_depth=self.config.lgbm_max_depth,
                learning_rate=self.config.lgbm_learning_rate,
                reg_alpha=self.config.lgbm_reg_alpha,
                reg_lambda=self.config.lgbm_reg_lambda,
                min_child_samples=self.config.lgbm_min_child_samples,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1,
                importance_type='gain'
            )
        
        elif self.config.model_type == MetaModelType.EXTRA_TREES:
            return ExtraTreesClassifier(
                n_estimators=self.config.et_n_estimators,
                max_depth=self.config.et_max_depth,
                min_samples_leaf=self.config.et_min_samples_leaf,
                max_features=self.config.et_max_features,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        elif self.config.model_type == MetaModelType.LINEAR:
            return LogisticRegression(
                C=self.config.linear_C,
                solver=self.config.linear_solver,
                max_iter=1000,
                random_state=self.config.random_state
            )
        
        raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def train_walk_forward(
        self,
        base_predictions: pd.DataFrame,
        disagreement_features: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        embargo_periods: int = 10
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Train using walk-forward validation with OOF predictions.
        
        Args:
            base_predictions: DataFrame with base model OOF predictions
            disagreement_features: DataFrame with disagreement features
            y: Target Series
            n_splits: Number of walk-forward splits
            embargo_periods: Embargo periods between train and val
            
        Returns:
            (OOF predictions Series, training metrics dict)
        """
        start_time = time.time()
        
        # Combine features
        if self.config.model_type == MetaModelType.SIMPLE_AVERAGE:
            # Simple average: just average base predictions
            X = base_predictions
            self.feature_names = list(base_predictions.columns)
        else:
            # Other models: use base predictions + disagreement features
            X = pd.concat([base_predictions, disagreement_features], axis=1)
            self.feature_names = list(X.columns)
        
        n_samples = len(X)
        
        # Find valid samples (where we have OOF predictions from Layer 1)
        # Fill NaN with forward fill first, then check
        X_filled = X.ffill().fillna(0)
        valid_mask = ~y.isna()
        valid_mask.iloc[:self.burn_in_periods] = False
        valid_indices = np.where(valid_mask)[0]
        
        # Use filled features for training
        X = X_filled
        
        tprint_info(f"[{self.config.name}] Valid samples: {len(valid_indices)}/{n_samples}")
        
        # Initialize OOF predictions
        oof_predictions = np.full(n_samples, np.nan)
        fold_metrics: List[Dict[str, float]] = []
        
        # Walk-forward splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(valid_indices)):
            train_idx = valid_indices[train_idx]
            val_idx = valid_indices[val_idx]
            
            # Apply embargo
            if embargo_periods > 0 and len(train_idx) > embargo_periods:
                train_idx = train_idx[:-embargo_periods]
            
            tprint_info(f"[{self.config.name}] Fold {fold_idx + 1}/{n_splits}: "
                       f"Train={len(train_idx)}, Val={len(val_idx)}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            if self.config.model_type == MetaModelType.SIMPLE_AVERAGE:
                # Simple average
                fold_predictions = X_val.mean(axis=1).values
            else:
                # Scale features for linear model
                if self.config.model_type == MetaModelType.LINEAR:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                else:
                    X_train_scaled = X_train.values
                    X_val_scaled = X_val.values
                
                # Train model
                model = self._create_model()
                
                if self.config.model_type == MetaModelType.REGULARIZED_LGBM:
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(30, verbose=False)]
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    fold_predictions = model.predict_proba(X_val_scaled)[:, 1]
                else:
                    # Linear model may have decision_function
                    fold_predictions = model.predict_proba(X_val_scaled)[:, 1]
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    imp_dict = dict(zip(self.feature_names, model.feature_importances_))
                    self.fold_feature_importances.append(imp_dict)
                elif hasattr(model, 'coef_'):
                    coef = model.coef_.ravel()
                    imp_dict = dict(zip(self.feature_names, np.abs(coef)))
                    self.fold_feature_importances.append(imp_dict)
                
                # Save model from last fold
                if fold_idx == n_splits - 1:
                    self.model = model
                    if self.config.model_type == MetaModelType.LINEAR:
                        self.scaler = scaler
            
            oof_predictions[val_idx] = fold_predictions
            
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
                "n_train": len(train_idx),
                "n_val": len(val_idx)
            })
        
        training_time = time.time() - start_time
        
        # Create OOF Series
        self.oof_predictions = pd.Series(
            oof_predictions, index=X.index, 
            name=f"{self.config.name}_prob"
        )
        
        # Check feature stability (top 3 features consistent across folds)
        feature_stability = self._check_feature_stability()
        
        # Aggregate training metrics
        self.training_metrics = {
            "model_name": self.config.name,
            "model_type": self.config.model_type.value,
            "n_splits": n_splits,
            "mean_auc": np.mean([m["auc"] for m in fold_metrics]),
            "std_auc": np.std([m["auc"] for m in fold_metrics]),
            "mean_logloss": np.mean([m["logloss"] for m in fold_metrics]),
            "fold_metrics": fold_metrics,
            "training_time_sec": training_time,
            "n_features": len(self.feature_names),
            "n_valid_samples": len(valid_indices),
            "feature_stability": feature_stability
        }
        
        tprint_success(f"[{self.config.name}] Training complete: "
                      f"AUC={self.training_metrics['mean_auc']:.4f} ± {self.training_metrics['std_auc']:.4f}")
        
        return self.oof_predictions, self.training_metrics
    
    def _check_feature_stability(self) -> Dict[str, Any]:
        """Check if top features are consistent across folds."""
        if len(self.fold_feature_importances) < 2:
            return {"consistent": True, "top_features": [], "stability_score": 1.0}
        
        # Get top 5 features from each fold
        top_features_per_fold = []
        for imp_dict in self.fold_feature_importances:
            sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            top_5 = [f[0] for f in sorted_features[:5]]
            top_features_per_fold.append(set(top_5))
        
        # Check overlap
        first_fold_top5 = top_features_per_fold[0]
        consistent_features = first_fold_top5.copy()
        
        for fold_top5 in top_features_per_fold[1:]:
            consistent_features = consistent_features.intersection(fold_top5)
        
        # Top 3 should be consistent
        all_top3 = []
        for imp_dict in self.fold_feature_importances:
            sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            top_3 = [f[0] for f in sorted_features[:3]]
            all_top3.append(set(top_3))
        
        first_top3 = all_top3[0]
        consistent_top3 = first_top3.copy()
        for top3 in all_top3[1:]:
            consistent_top3 = consistent_top3.intersection(top3)
        
        stability_score = len(consistent_features) / 5 if len(top_features_per_fold) > 0 else 0
        
        return {
            "consistent": len(consistent_top3) >= 2,
            "top_features": list(consistent_features),
            "consistent_top3": list(consistent_top3),
            "stability_score": stability_score
        }
    
    def predict(self, base_predictions: pd.DataFrame, disagreement_features: pd.DataFrame) -> np.ndarray:
        """
        Predict using trained meta model.
        
        Args:
            base_predictions: DataFrame with base model predictions
            disagreement_features: DataFrame with disagreement features
            
        Returns:
            Probability predictions
        """
        if self.config.model_type == MetaModelType.SIMPLE_AVERAGE:
            return base_predictions.mean(axis=1).values
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = pd.concat([base_predictions, disagreement_features], axis=1)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict(X_scaled)


class Layer2Orchestrator:
    """
    Orchestrates training of multiple Layer 2 meta models.
    
    Manages:
    - Multiple meta model types (average, LGBM, ExtraTrees, linear)
    - Disagreement feature generation
    - OOF prediction aggregation
    - Comparison against base models
    - Metrics reporting
    """
    
    def __init__(
        self,
        meta_model_configs: Optional[List[MetaModelConfig]] = None,
        reporter: Optional[MultiLayerMetricsReporter] = None,
        symbol: str = "UNKNOWN",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            meta_model_configs: List of meta model configurations
            reporter: Metrics reporter instance
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
        """
        self.meta_model_configs = meta_model_configs or DEFAULT_META_CONFIGS
        self.reporter = reporter or MultiLayerMetricsReporter()
        
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.direction = direction
        
        self.disagreement_generator = DisagreementFeatureGenerator()
        self.trainers: Dict[str, MetaModelTrainer] = {}
        self.oof_predictions: Dict[str, pd.Series] = {}
        self.all_metrics: List[LayerMetrics] = []
        self.disagreement_features: Optional[pd.DataFrame] = None
        
        # For comparison
        self.best_base_model_auc: float = 0.0
        self.best_base_model_name: str = ""
    
    def set_baseline_performance(self, base_metrics: List[LayerMetrics]) -> None:
        """
        Set baseline performance from Layer 1 for comparison.
        
        Args:
            base_metrics: List of Layer 1 metrics
        """
        if not base_metrics:
            return
        
        best = max(base_metrics, key=lambda m: m.predictive.auc_roc)
        self.best_base_model_auc = best.predictive.auc_roc
        self.best_base_model_name = best.model_name
        
        tprint_info(f"📊 Best base model: {self.best_base_model_name} (AUC={self.best_base_model_auc:.4f})")
    
    def train_all_models(
        self,
        base_predictions: pd.DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None,
        n_splits: int = 5
    ) -> Dict[str, pd.Series]:
        """
        Train all meta models and collect OOF predictions.
        
        Args:
            base_predictions: DataFrame with base model OOF predictions
            y: Target Series (binary labels)
            returns: Optional returns for trading metrics
            n_splits: Number of walk-forward splits
            
        Returns:
            Dict of model_name -> OOF predictions Series
        """
        tprint_info("=" * 80)
        tprint_info("LAYER 2: TRAINING META MODELS")
        tprint_info("=" * 80)
        
        # Generate disagreement features
        tprint_info("\n🔧 Generating disagreement features...")
        self.disagreement_features = self.disagreement_generator.generate(base_predictions)
        tprint_success(f"✅ Generated {self.disagreement_features.shape[1]} disagreement features")
        
        for config in self.meta_model_configs:
            tprint_info(f"\n📊 Training: {config.name} ({config.model_type.value})")
            tprint_info("-" * 40)
            
            # Create trainer
            trainer = MetaModelTrainer(
                config=config,
                burn_in_periods=100
            )
            
            # Train and get OOF predictions
            oof_preds, training_metrics = trainer.train_walk_forward(
                base_predictions,
                self.disagreement_features,
                y,
                n_splits=n_splits
            )
            
            self.trainers[config.name] = trainer
            self.oof_predictions[config.name] = oof_preds
            
            # Compute comprehensive metrics
            layer_metrics = self._compute_layer_metrics(
                config.name,
                config.model_type.value,
                oof_preds,
                y,
                returns,
                training_metrics
            )
            
            self.all_metrics.append(layer_metrics)
            self.reporter.record_metrics(layer_metrics)
            
            # Generate individual markdown report
            report_path = self.reporter.output_dir / f"L2_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            generate_layer_markdown_report(layer_metrics, str(report_path))
        
        # Generate comparison summary
        self._generate_comparison_summary()
        
        return self.oof_predictions
    
    def _compute_layer_metrics(
        self,
        model_name: str,
        model_type: str,
        oof_preds: pd.Series,
        y: pd.Series,
        returns: Optional[pd.Series],
        training_metrics: Dict[str, Any]
    ) -> LayerMetrics:
        """Compute comprehensive Layer 2 metrics."""
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
        
        # Feature stability
        feature_stability = training_metrics.get("feature_stability", {})
        stability = StabilityMetrics(
            top_5_features_consistent=feature_stability.get("consistent", False),
            feature_importance_shift=1 - feature_stability.get("stability_score", 0)
        )
        
        # Create LayerMetrics
        metrics = LayerMetrics(
            model_name=model_name,
            layer="L2_meta",
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            direction=self.direction,
            model_type=model_type,
            calibration=calibration,
            stability=stability,
            trading=trading,
            risk=risk,
            predictive=predictive,
            activity=ActivityMetrics(),
            diversity=DiversityMetrics(),
            gate=GateMetrics(),
            n_samples=int(valid_mask.sum()),
            n_features=training_metrics.get("n_features", 0),
            training_duration_sec=training_metrics.get("training_time_sec", 0),
            notes=f"AUC: {training_metrics.get('mean_auc', 0):.4f}, "
                  f"vs Best Base: {predictive.auc_roc - self.best_base_model_auc:+.4f}"
        )
        
        return metrics
    
    def _generate_comparison_summary(self) -> None:
        """Generate comparison summary across all meta models."""
        tprint_info("\n" + "=" * 80)
        tprint_info("LAYER 2 MODEL COMPARISON")
        tprint_info("=" * 80)
        
        # Sort by AUC
        sorted_metrics = sorted(self.all_metrics, key=lambda m: m.predictive.auc_roc, reverse=True)
        
        tprint_info(f"\n{'Model':<25} {'AUC':<10} {'ECE':<10} {'Brier':<10} {'IC':<10} {'vs Base':<10}")
        tprint_info("-" * 75)
        
        for m in sorted_metrics:
            delta = m.predictive.auc_roc - self.best_base_model_auc
            delta_str = f"{delta:+.4f}"
            tprint_info(f"{m.model_name:<25} {m.predictive.auc_roc:<10.4f} "
                       f"{m.calibration.ece:<10.4f} {m.calibration.brier_score:<10.4f} "
                       f"{m.predictive.information_coefficient:<10.4f} {delta_str:<10}")
        
        # Identify best model
        best = sorted_metrics[0]
        tprint_success(f"\n🏆 Best Meta Model: {best.model_name} (AUC={best.predictive.auc_roc:.4f})")
        
        # Check success criteria
        self._check_success_criteria(best)
    
    def _check_success_criteria(self, best: LayerMetrics) -> None:
        """Check Layer 2 success criteria."""
        tprint_info("\n📋 Layer 2 Success Criteria Check:")
        
        # IC Improvement
        ic_improvement = (best.predictive.information_coefficient - self.best_base_model_auc) / max(0.01, self.best_base_model_auc)
        ic_pass = ic_improvement >= 0.20
        tprint_info(f"   IC Improvement: {ic_improvement*100:.1f}% (target: +20%) {'✅' if ic_pass else '❌'}")
        
        # ECE
        ece_pass = best.calibration.ece < 0.05
        tprint_info(f"   ECE: {best.calibration.ece:.4f} (target: <0.05) {'✅' if ece_pass else '❌'}")
        
        # Profit Factor
        pf_pass = best.trading.profit_factor > 1.8
        tprint_info(f"   Profit Factor: {best.trading.profit_factor:.4f} (target: >1.8) {'✅' if pf_pass else '❌'}")
        
        # Feature Stability
        fs_pass = best.stability.top_5_features_consistent
        tprint_info(f"   Top 3 Features Consistent: {fs_pass} {'✅' if fs_pass else '❌'}")
    
    def get_best_model(self) -> Tuple[str, MetaModelTrainer]:
        """Get the best performing meta model."""
        if not self.all_metrics:
            raise ValueError("No models trained yet")
        
        best = max(self.all_metrics, key=lambda m: m.predictive.auc_roc)
        return best.model_name, self.trainers[best.model_name]
    
    def get_combined_oof_predictions(self) -> pd.DataFrame:
        """Get all OOF predictions as a DataFrame."""
        return pd.DataFrame(self.oof_predictions)
    
    def get_disagreement_features(self) -> pd.DataFrame:
        """Get the generated disagreement features."""
        return self.disagreement_features
