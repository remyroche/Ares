"""
Causal Specialists Module

Implements causal specialists as causal parents in the structural causal model.
Manages specialist predictions, surprise detection, and coordination.

Key Features:
1. Causal specialists as causal parents
2. Specialist prediction tracking
3. Surprise detection and management
4. Specialist coordination and consensus
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from dataclasses import dataclass

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class CausalSpecialist:
    """
    Individual causal specialist representing a causal parent.
    
    Each specialist focuses on a specific causal relationship and
    provides predictions for that domain.
    """
    
    def __init__(
        self,
        name: str,
        causal_parent: str,
        causal_child: str,
        model_type: str = "linear",
        prediction_window: int = 1,
        confidence_threshold: float = 0.5,
        verbose: bool = True,
        asset_id_col: str = "asset_id",
        asset_id_onehot_max: Optional[int] = None,
        asset_id_use_ridge: bool = True,
        asset_id_ridge_alpha: float = 1.0,
        asset_id_other_label: str = "OTHER",
        asset_residual_scale: float = 0.3,
        asset_residual_ridge_multiplier: float = 5.0,
        asset_embedding_dim: int = 0,
        asset_embedding_l2: float = 1.0,
        asset_embedding_l2_multiplier: float = 5.0,
        asset_embedding_max_norm: Optional[float] = None,
        asset_embedding_dropout: float = 0.0,
        asset_embedding_lr: float = 0.05,
        asset_embedding_epochs: int = 50,
        asset_embedding_batch_size: int = 2048,
        asset_embedding_sample_limit: int = 50000,
        asset_embedding_interaction: bool = True,
        asset_embedding_random_state: int = 42,
    ):
        """
        Initialize causal specialist.
        
        Args:
            name: Specialist name
            causal_parent: Parent variable in causal relationship
            causal_child: Child variable in causal relationship
            model_type: Type of model to use
            prediction_window: Window for predictions
            confidence_threshold: Threshold for confidence
            verbose: Whether to print progress information
        """
        self.name = name
        self.causal_parent = causal_parent
        self.causal_child = causal_child
        self.model_type = model_type
        self.prediction_window = prediction_window
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.asset_id_col = asset_id_col
        if asset_id_onehot_max in (None, 0, "none", "None"):
            self.asset_id_onehot_max = None
        else:
            self.asset_id_onehot_max = int(asset_id_onehot_max)
        self.asset_id_use_ridge = bool(asset_id_use_ridge)
        self.asset_id_ridge_alpha = float(asset_id_ridge_alpha)
        self.asset_id_other_label = asset_id_other_label
        self.asset_residual_scale = float(asset_residual_scale)
        self.asset_residual_ridge_multiplier = float(asset_residual_ridge_multiplier)
        self.asset_embedding_dim = int(asset_embedding_dim)
        self.asset_embedding_l2 = float(asset_embedding_l2)
        self.asset_embedding_l2_multiplier = float(asset_embedding_l2_multiplier)
        self.asset_embedding_max_norm = (
            float(asset_embedding_max_norm) if asset_embedding_max_norm is not None else None
        )
        self.asset_embedding_dropout = float(asset_embedding_dropout)
        self.asset_embedding_lr = float(asset_embedding_lr)
        self.asset_embedding_epochs = int(asset_embedding_epochs)
        self.asset_embedding_batch_size = int(asset_embedding_batch_size)
        self.asset_embedding_sample_limit = int(asset_embedding_sample_limit)
        self.asset_embedding_interaction = bool(asset_embedding_interaction)
        self.asset_embedding_random_state = int(asset_embedding_random_state)
        
        # Model and data storage
        self.model = None
        self.predictions_ = None
        self.prediction_errors_ = None
        self.confidence_scores_ = None
        self.surprise_events_ = None
        
        # Performance metrics
        self.performance_metrics_ = {}

        # Asset encoding cache
        self._asset_id_top_values: Optional[List[str]] = None
        self._asset_id_feature_cols: Optional[List[str]] = None
        self._asset_features_used: bool = False
        self._feature_columns_: Optional[List[str]] = None
        self._base_feature_columns_: Optional[List[str]] = None
        self._asset_residual_model: Optional[Any] = None
        self._asset_embedding_matrix_: Optional[np.ndarray] = None
        self._asset_embedding_weights_: Optional[np.ndarray] = None
        self._asset_embedding_bias_: float = 0.0
        self._asset_embedding_index_: Optional[Dict[str, int]] = None
        self._asset_embedding_enabled_: bool = False
        
    def _create_model(self) -> Any:
        """
        Create prediction model based on type.
        
        Returns:
            Model instance
        """
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=self.asset_id_ridge_alpha)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=50, random_state=42, max_depth=5
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _extract_asset_series(self, X: pd.DataFrame) -> Optional[pd.Series]:
        if self.asset_id_col in X.columns:
            return X[self.asset_id_col]
        if isinstance(X.index, pd.MultiIndex) and self.asset_id_col in X.index.names:
            return pd.Series(X.index.get_level_values(self.asset_id_col), index=X.index)
        return None

    def _make_asset_col_name(self, asset_value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(asset_value))
        return f"{self.asset_id_col}_{safe}"

    def _encode_asset_features(self, X: pd.DataFrame, fit: bool) -> Optional[pd.DataFrame]:
        asset_series = self._extract_asset_series(X)
        if asset_series is None:
            return None

        asset_series = asset_series.astype(str)
        if asset_series.nunique(dropna=False) < 2:
            return None

        if fit or self._asset_id_top_values is None:
            if self.asset_id_onehot_max is None or self.asset_id_onehot_max <= 0:
                self._asset_id_top_values = asset_series.value_counts(dropna=False).index.tolist()
            else:
                self._asset_id_top_values = (
                    asset_series.value_counts(dropna=False)
                    .head(self.asset_id_onehot_max)
                    .index
                    .tolist()
                )
            self._asset_id_feature_cols = [
                self._make_asset_col_name(asset) for asset in self._asset_id_top_values
            ]
            if asset_series.nunique(dropna=False) > len(self._asset_id_top_values):
                self._asset_id_feature_cols.append(
                    self._make_asset_col_name(self.asset_id_other_label)
                )

        top_assets = self._asset_id_top_values or []
        use_other = asset_series.nunique(dropna=False) > len(top_assets)
        bucketed = asset_series.where(asset_series.isin(top_assets), other=self.asset_id_other_label)
        dummies = pd.get_dummies(bucketed, dtype=float)

        rename_map = {asset: self._make_asset_col_name(asset) for asset in top_assets}
        if use_other:
            rename_map[self.asset_id_other_label] = self._make_asset_col_name(self.asset_id_other_label)
        dummies = dummies.rename(columns=rename_map)

        feature_cols = self._asset_id_feature_cols or list(rename_map.values())
        for col in feature_cols:
            if col not in dummies.columns:
                dummies[col] = 0.0
        dummies = dummies.reindex(columns=feature_cols, fill_value=0.0)
        return dummies

    def _build_base_matrix(
        self, X: pd.DataFrame, fit: bool
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        if self.causal_parent in X.columns:
            parent_col = X[self.causal_parent]
            if parent_col.dtype == 'object' or str(parent_col.dtype) == 'string':
                return None, f"non-numeric parent ({self.causal_parent})"
            base_df = X[[self.causal_parent]]
        else:
            base_df = X.select_dtypes(include=[np.number]).copy()
            if self.asset_id_col in base_df.columns:
                base_df = base_df.drop(columns=[self.asset_id_col])
            if base_df.empty:
                return None, "no numeric columns"

        base_df = base_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if fit:
            self._base_feature_columns_ = list(base_df.columns)
        elif self._base_feature_columns_ is not None:
            base_df = base_df.reindex(columns=self._base_feature_columns_, fill_value=0.0)

        return base_df, None

    def _compute_embedding_base_signal(self, base_df: pd.DataFrame) -> np.ndarray:
        if not self.asset_embedding_interaction:
            return np.ones(len(base_df), dtype=float)
        if base_df.shape[1] == 1:
            return base_df.iloc[:, 0].to_numpy(dtype=float)
        return base_df.mean(axis=1).to_numpy(dtype=float)

    def _fit_asset_embedding_residual(
        self,
        base_df: pd.DataFrame,
        asset_series: pd.Series,
        y_resid: np.ndarray,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[np.ndarray]:
        if self.asset_embedding_dim <= 0:
            return None

        asset_values = asset_series.astype(str).fillna(self.asset_id_other_label)
        unique_assets = pd.unique(asset_values)
        if len(unique_assets) < 2:
            return None

        asset_to_idx = {asset: idx for idx, asset in enumerate(unique_assets)}
        asset_idx = asset_values.map(asset_to_idx).to_numpy(dtype=int)

        rng = np.random.default_rng(self.asset_embedding_random_state)
        n_assets = len(unique_assets)
        dim = int(self.asset_embedding_dim)

        emb = rng.normal(scale=0.01, size=(n_assets, dim))
        weights = rng.normal(scale=0.01, size=dim)
        bias = 0.0

        base_signal = self._compute_embedding_base_signal(base_df)
        base_signal = np.asarray(base_signal, dtype=float)
        y_target = np.asarray(y_resid, dtype=float)

        if sample_weights is not None:
            weights_arr = sample_weights.reindex(base_df.index).fillna(1.0).to_numpy(dtype=float)
        else:
            weights_arr = np.ones(len(base_df), dtype=float)

        n_samples = len(base_df)
        sample_limit = max(0, int(self.asset_embedding_sample_limit))
        if sample_limit and n_samples > sample_limit:
            sample_idx = rng.choice(n_samples, size=sample_limit, replace=False)
        else:
            sample_idx = np.arange(n_samples)

        epochs = max(1, int(self.asset_embedding_epochs))
        batch_size = max(32, int(self.asset_embedding_batch_size))
        lr = float(self.asset_embedding_lr)
        l2_w = float(self.asset_embedding_l2)
        l2_e = float(self.asset_embedding_l2) * float(self.asset_embedding_l2_multiplier)
        dropout = float(self.asset_embedding_dropout)
        max_norm = self.asset_embedding_max_norm

        for _ in range(epochs):
            rng.shuffle(sample_idx)
            for start in range(0, len(sample_idx), batch_size):
                batch = sample_idx[start:start + batch_size]
                a_idx = asset_idx[batch]
                z = base_signal[batch]
                y_b = y_target[batch]
                w_b = weights_arr[batch]

                emb_batch = emb[a_idx]
                if dropout > 0:
                    mask = rng.binomial(1, 1 - dropout, size=emb_batch.shape)
                    emb_batch = emb_batch * mask / max(1e-9, 1 - dropout)

                pred = (emb_batch @ weights) * z + bias
                err = pred - y_b
                weight_norm = np.sum(w_b) + 1e-12
                err = err * (w_b / weight_norm)

                grad_w = 2.0 * np.sum((err * z)[:, None] * emb_batch, axis=0) + 2.0 * l2_w * weights
                grad_b = 2.0 * np.sum(err)

                grad_e = (err * z)[:, None] * weights[None, :]
                grad_e = 2.0 * grad_e
                if l2_e > 0:
                    grad_e = grad_e + 2.0 * l2_e * emb_batch

                np.add.at(emb, a_idx, -lr * grad_e)
                weights = weights - lr * grad_w
                bias = bias - lr * grad_b

                if max_norm is not None:
                    norms = np.linalg.norm(emb, axis=1)
                    too_big = norms > max_norm
                    if np.any(too_big):
                        emb[too_big] = emb[too_big] / norms[too_big][:, None] * max_norm

        self._asset_embedding_matrix_ = emb
        self._asset_embedding_weights_ = weights
        self._asset_embedding_bias_ = float(bias)
        self._asset_embedding_index_ = asset_to_idx
        self._asset_embedding_enabled_ = True
        return self._predict_asset_embedding_residual(base_df, asset_series)

    def _predict_asset_embedding_residual(
        self, base_df: pd.DataFrame, asset_series: pd.Series
    ) -> np.ndarray:
        if not self._asset_embedding_enabled_:
            return np.zeros(len(base_df), dtype=float)
        if self._asset_embedding_matrix_ is None or self._asset_embedding_weights_ is None:
            return np.zeros(len(base_df), dtype=float)

        asset_values = asset_series.astype(str).fillna(self.asset_id_other_label)
        asset_to_idx = self._asset_embedding_index_ or {}
        idx = asset_values.map(asset_to_idx).fillna(-1).to_numpy(dtype=int)
        emb = self._asset_embedding_matrix_
        weights = self._asset_embedding_weights_

        base_signal = self._compute_embedding_base_signal(base_df)
        base_signal = np.asarray(base_signal, dtype=float)

        emb_rows = np.zeros((len(base_df), emb.shape[1]), dtype=float)
        valid = idx >= 0
        if np.any(valid):
            emb_rows[valid] = emb[idx[valid]]

        return (emb_rows @ weights) * base_signal + self._asset_embedding_bias_

    def _build_feature_matrix(
        self, X: pd.DataFrame, fit: bool
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        base_df, error = self._build_base_matrix(X, fit=fit)
        if base_df is None:
            return None, error

        asset_df = self._encode_asset_features(X, fit=fit)
        self._asset_features_used = asset_df is not None and not asset_df.empty

        if asset_df is not None and not asset_df.empty:
            feature_df = pd.concat([base_df, asset_df], axis=1)
        else:
            feature_df = base_df

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if fit:
            self._feature_columns_ = list(feature_df.columns)
        elif self._feature_columns_ is not None:
            feature_df = feature_df.reindex(columns=self._feature_columns_, fill_value=0.0)

        return feature_df, None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Fit the specialist model.
        
        Args:
            X: Feature matrix (parent variable)
            y: Target vector (child variable)
            sample_weights: Optional sample weights
            
        Returns:
            Training metrics
        """
        try:
            if self.verbose:
                tprint_info(f"🧠 Training Specialist: {self.name}")
            
            # Check if y is numeric
            if y.dtype == 'object' or str(y.dtype) == 'string':
                tprint_warning(f"   ⚠️ Skipping {self.name}: target is non-numeric")
                self.is_fitted_ = False
                return {"error": "non-numeric target", "skipped": True}

            base_df, error = self._build_base_matrix(X, fit=True)
            if base_df is None:
                tprint_warning(f"   ⚠️ Skipping {self.name}: {error}")
                self.is_fitted_ = False
                return {"error": error or "invalid features", "skipped": True}

            self._asset_residual_model = None
            self._asset_embedding_enabled_ = False

            if self.model_type == "ridge":
                self.model = Ridge(alpha=self.asset_id_ridge_alpha)
            else:
                self.model = self._create_model()
            
            # Handle y fillna for numeric only
            if pd.api.types.is_numeric_dtype(y):
                y_train = y.replace([np.inf, -np.inf], np.nan)
                if y_train.isna().all():
                    tprint_warning(f"   ⚠️ Skipping {self.name}: y is all NaN/inf")
                    self.is_fitted_ = False
                    return {"error": "y all nan", "skipped": True}
                y_mean = y_train.mean()
                y_train = y_train.fillna(y_mean)
            else:
                tprint_warning(f"   ⚠️ Skipping {self.name}: y is not numeric")
                self.is_fitted_ = False
                return {"error": "y not numeric", "skipped": True}
            
            # Fit global model
            sample_weight = None
            if sample_weights is not None:
                sample_weight = sample_weights.reindex(base_df.index).fillna(0).to_numpy()
            if sample_weight is not None and hasattr(self.model, "fit"):
                self.model.fit(base_df, y_train, sample_weight=sample_weight)
            else:
                self.model.fit(base_df, y_train)
            self.is_fitted_ = True

            global_pred = self.model.predict(base_df)
            residual_target = y_train.to_numpy(dtype=float) - global_pred

            asset_series = self._extract_asset_series(X)
            asset_residual_pred = np.zeros(len(base_df), dtype=float)
            if asset_series is not None and asset_series.nunique(dropna=False) > 1:
                if self.asset_embedding_dim > 0:
                    asset_residual_pred = (
                        self._fit_asset_embedding_residual(
                            base_df, asset_series, residual_target, sample_weights
                        )
                        or asset_residual_pred
                    )
                else:
                    asset_df = self._encode_asset_features(X, fit=True)
                    if asset_df is not None and not asset_df.empty:
                        ridge_alpha = self.asset_id_ridge_alpha * self.asset_residual_ridge_multiplier
                        self._asset_residual_model = Ridge(alpha=ridge_alpha)
                        if sample_weight is not None:
                            self._asset_residual_model.fit(
                                asset_df, residual_target, sample_weight=sample_weight
                            )
                        else:
                            self._asset_residual_model.fit(asset_df, residual_target)
                        asset_residual_pred = self._asset_residual_model.predict(asset_df)

            predictions = global_pred + self.asset_residual_scale * asset_residual_pred
            
            # Compute prediction errors
            errors = y_train - predictions
            
            # Compute confidence scores (simplified)
            if hasattr(self.model, 'predict'):
                # Use prediction variance as confidence proxy
                confidence = 1.0 / (1.0 + np.var(errors))
            else:
                confidence = 0.5
            
            # Store results
            self.predictions_ = pd.Series(predictions, index=base_df.index)
            self.prediction_errors_ = pd.Series(errors, index=base_df.index)
            self.confidence_scores_ = pd.Series(
                np.full(len(predictions), confidence),
                index=base_df.index
            )
            
            # Compute performance metrics
            mse = mean_squared_error(y_train, predictions)
            mae = mean_absolute_error(y_train, predictions)
            try:
                r2 = self.model.score(base_df, y_train)
            except Exception:
                r2 = float("nan")
            
            self.performance_metrics_ = {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "n_samples": len(base_df),
                "mean_error": errors.mean(),
                "std_error": errors.std()
            }
            
            if self.verbose:
                tprint_success(f"   ✅ Training complete:")
                tprint_info(f"      - MSE: {mse:.6f}")
                tprint_info(f"      - R²: {r2:.4f}")
                tprint_info(f"      - Samples: {len(base_df)}")
            
            return self.performance_metrics_
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist training failed: {e}")
            raise
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = True
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate predictions.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions and optionally confidence scores
        """
        try:
            if self.model is None or (hasattr(self, 'is_fitted_') and not self.is_fitted_):
                return pd.Series(np.nan, index=X.index)
            
            base_df, error = self._build_base_matrix(X, fit=False)
            if base_df is None:
                return pd.Series(np.nan, index=X.index)
            
            # Check if model is fitted before predicting
            from sklearn.utils.validation import check_is_fitted
            try:
                check_is_fitted(self.model)
            except Exception:
                tprint_warning(f"⚠️ Model for {self.name} is not fitted, returning empty predictions")
                return pd.Series(np.nan, index=base_df.index)
            
            # Generate predictions
            global_pred = self.model.predict(base_df)
            asset_residual_pred = np.zeros(len(base_df), dtype=float)
            asset_series = self._extract_asset_series(X)
            if asset_series is not None and asset_series.nunique(dropna=False) > 1:
                if self._asset_embedding_enabled_:
                    asset_residual_pred = self._predict_asset_embedding_residual(base_df, asset_series)
                elif self._asset_residual_model is not None:
                    asset_df = self._encode_asset_features(X, fit=False)
                    if asset_df is not None and not asset_df.empty:
                        asset_residual_pred = self._asset_residual_model.predict(asset_df)

            predictions = global_pred + self.asset_residual_scale * asset_residual_pred
            
            # Store predictions
            self.predictions_ = pd.Series(predictions, index=base_df.index)
            
            if not return_confidence:
                return self.predictions_
            
            # Generate confidence scores (simplified)
            if hasattr(self.model, 'predict'):
                # Use historical error variance
                if self.prediction_errors_ is not None:
                    error_var = self.prediction_errors_.var()
                    confidence = 1.0 / (1.0 + error_var)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            confidence_scores = pd.Series(
                np.full(len(predictions), confidence),
                index=base_df.index
            )
            
            self.confidence_scores_ = confidence_scores
            
            return self.predictions_, self.confidence_scores_
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def detect_surprise_events(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        surprise_threshold: float = 1.8
    ) -> pd.Series:
        """
        Detect surprise events for this specialist.
        
        Args:
            X: Feature matrix
            y: True values
            surprise_threshold: Threshold for surprise detection
            
        Returns:
            Surprise event indicators
        """
        try:
            if self.verbose:
                tprint_info(f"🔍 Detecting Surprise Events: {self.name}")
            
            # Generate predictions
            predictions, confidence = self.predict(X, return_confidence=True)
            
            # Compute prediction errors
            errors = y - predictions
            
            # Detect surprise events using standardized rolling quantile logic
            # Use 'z_scores' proxy or raw errors. But util expects a series.
            # Best to use ABSOLUTE ERRORS for magnitude detection
            abs_errors = errors.abs()
            
            # Using Detection Util (Standardized)
            from .detection_utils import detect_rolling_quantile_surprises
            
            # Use window=20 as per original method, or update to standard?
            # User asked to "Verify if this same approach can be applied... std dev of xx-bar rolling window"
            # Original code used window=20. We can stick to 20 or bump to 100 for robustness.
            # Let's use window=100 (more robust) but min_periods=20.
            details = detect_rolling_quantile_surprises(
                abs_errors, 
                window=100, 
                quantiles=(0.96, 0.98), # Standardized 96/98
                min_periods=20, 
                return_details=True
            )
            
            # Surprise event = Level >= 2.0
            surprise_events = details['level'] >= 2.0
            
            # Store results
            self.surprise_events_ = surprise_events
            if 'weight' in details:
                self.surprise_weights_ = details['weight']
            
            if self.verbose:
                n_surprises = surprise_events.sum()
                tprint_success(f"   ✅ Surprise detection complete (Standardized Quantiles):")
                tprint_info(f"      - Surprise events: {n_surprises}")
                tprint_info(f"      - Surprise rate: {n_surprises/len(surprise_events):.2%}")
            
            return surprise_events
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Surprise detection failed: {e}")
            return pd.Series(False, index=X.index)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of specialist performance.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "name": self.name,
            "causal_parent": self.causal_parent,
            "causal_child": self.causal_child,
            "model_type": self.model_type,
            "is_fitted": self.model is not None,
            "has_predictions": self.predictions_ is not None,
            "has_surprise_events": self.surprise_events_ is not None
        }
        
        if self.performance_metrics_:
            summary.update(self.performance_metrics_)
        
        if self.surprise_events_ is not None:
            summary["surprise_events_count"] = self.surprise_events_.sum()
            summary["surprise_rate"] = self.surprise_events_.mean()
        
        return summary

class CausalSpecialistManager:
    """
    Manages a collection of causal specialists.
    
    Coordinates multiple specialists, tracks their performance,
    and manages consensus and surprise detection.
    """
    
    def __init__(
        self,
        specialists: Optional[List[CausalSpecialist]] = None,
        consensus_threshold: float = 0.6,
        surprise_aggregation: str = "majority",
        verbose: bool = True
    ):
        """
        Initialize Causal Specialist Manager.
        
        Args:
            specialists: List of specialists
            consensus_threshold: Threshold for consensus
            surprise_aggregation: Method for aggregating surprises
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.consensus_threshold = consensus_threshold
        self.surprise_aggregation = surprise_aggregation
        
        # Specialist storage
        self.specialists = specialists or []
        self.specialist_dict_ = {}
        
        # Performance tracking
        self.consensus_predictions_ = None
        self.aggregated_surprises_ = None
        self.manager_metrics_ = {}
        
        # Initialize specialist dictionary
        for specialist in self.specialists:
            self.specialist_dict_[specialist.name] = specialist
    
    def add_specialist(self, specialist: CausalSpecialist) -> None:
        """
        Add a specialist to the manager.
        
        Args:
            specialist: Specialist to add
        """
        self.specialists.append(specialist)
        self.specialist_dict_[specialist.name] = specialist
        
        if self.verbose:
            tprint_info(f"📝 Added specialist: {specialist.name}")
    
    def fit_all_specialists(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, pd.Series],
        sample_weights: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit all specialists.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of targets for each specialist
            sample_weights: Optional sample weights
            
        Returns:
            Training metrics for all specialists
        """
        try:
            if self.verbose:
                tprint_info("🚀 Training All Specialists...")
            
            training_metrics = {}
            
            for specialist in self.specialists:
                if specialist.name in y_dict:
                    y_target = y_dict[specialist.name]
                    weights = sample_weights.get(specialist.name) if sample_weights else None
                    
                    try:
                        metrics = specialist.fit(X, y_target, weights)
                        training_metrics[specialist.name] = metrics
                    except Exception as e:
                        if self.verbose:
                            tprint_warning(f"⚠️ Training failed for {specialist.name}: {e}")
                        training_metrics[specialist.name] = {"error": str(e)}
                else:
                    if self.verbose:
                        tprint_warning(f"⚠️ No target data for {specialist.name}")
            
            self.manager_metrics_["training_metrics"] = training_metrics
            
            if self.verbose:
                n_successful = sum(1 for m in training_metrics.values() if "error" not in m)
                tprint_success(f"✅ Training complete: {n_successful}/{len(self.specialists)} specialists")
            
            return training_metrics
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist training failed: {e}")
            raise
    
    def predict_all_specialists(
        self,
        X: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict[str, Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
        """
        Generate predictions from all specialists.
        
        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary of predictions from all specialists
        """
        try:
            if self.verbose:
                tprint_info("🔮 Generating Predictions from All Specialists...")
            
            predictions = {}
            
            for specialist in self.specialists:
                try:
                    pred = specialist.predict(X, return_confidence)
                    predictions[specialist.name] = pred
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"⚠️ Prediction failed for {specialist.name}: {e}")
            
            # Compute consensus predictions
            if predictions:
                self.consensus_predictions_ = self._compute_consensus(predictions)
            
            if self.verbose:
                tprint_success(f"✅ Predictions complete: {len(predictions)} specialists")
            
            return predictions
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist prediction failed: {e}")
            raise
    
    def detect_all_surprises(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, pd.Series],
        surprise_threshold: float = 1.8
    ) -> Dict[str, pd.Series]:
        """
        Detect surprise events from all specialists.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of true values
            surprise_threshold: Threshold for surprise detection
            
        Returns:
            Dictionary of surprise events from all specialists
        """
        try:
            if self.verbose:
                tprint_info("🚨 Detecting Surprise Events from All Specialists...")
            
            surprise_events = {}
            
            for specialist in self.specialists:
                if specialist.name in y_dict:
                    try:
                        surprises = specialist.detect_surprise_events(
                            X, y_dict[specialist.name], surprise_threshold
                        )
                        surprise_events[specialist.name] = surprises
                    except Exception as e:
                        if self.verbose:
                            tprint_warning(f"⚠️ Surprise detection failed for {specialist.name}: {e}")
            
            # Aggregate surprise events
            if surprise_events:
                self.aggregated_surprises_ = self._aggregate_surprises(surprise_events)
            
            if self.verbose:
                tprint_success(f"✅ Surprise detection complete: {len(surprise_events)} specialists")
            
            return surprise_events
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Surprise detection failed: {e}")
            raise
    
    def _compute_consensus(
        self,
        predictions: Dict[str, Union[pd.Series, Tuple[pd.Series, pd.Series]]]
    ) -> pd.DataFrame:
        """
        Compute consensus predictions from all specialists.
        
        Args:
            predictions: Dictionary of predictions
            
        Returns:
            Consensus predictions DataFrame
        """
        try:
            # Extract predictions (ignore confidence for consensus)
            pred_values = {}
            
            for name, pred in predictions.items():
                if isinstance(pred, tuple):
                    pred_values[name] = pred[0]
                else:
                    pred_values[name] = pred
            
            # Create DataFrame
            consensus_df = pd.DataFrame(pred_values)
            
            # Compute consensus statistics
            consensus_df["consensus_mean"] = consensus_df.mean(axis=1)
            consensus_df["consensus_std"] = consensus_df.std(axis=1)
            consensus_df["n_agreeing"] = (consensus_df.iloc[:, :-2] > 0).sum(axis=1)  # Assuming binary
            
            return consensus_df
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Consensus computation failed: {e}")
            return pd.DataFrame()
    
    def _aggregate_surprises(
        self,
        surprise_events: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Aggregate surprise events from all specialists.
        
        Args:
            surprise_events: Dictionary of surprise events
            
        Returns:
            Aggregated surprise events DataFrame
        """
        try:
            # Create DataFrame
            surprise_df = pd.DataFrame(surprise_events)
            
            # Aggregate based on method
            if self.surprise_aggregation == "majority":
                surprise_df["majority_surprise"] = (surprise_df.sum(axis=1) > len(surprise_df) / 2).astype(int)
            elif self.surprise_aggregation == "any":
                surprise_df["any_surprise"] = (surprise_df.sum(axis=1) > 0).astype(int)
            elif self.surprise_aggregation == "weighted":
                # Weight by specialist performance (simplified)
                weights = np.ones(len(surprise_df.columns))
                surprise_df["weighted_surprise"] = (surprise_df * weights).sum(axis=1) / weights.sum()
            else:
                surprise_df["sum_surprise"] = surprise_df.sum(axis=1)
            
            if 'weights' in kwargs:
                # Use provided weights (continuous severity)
                # Weighted Sum of Specialist Weights
                # If aggregation method supports it
                pass
            
            return surprise_df
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Surprise aggregation failed: {e}")
            return pd.DataFrame()
    
    def get_specialist_consensus(
        self,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get specialist consensus information.
        
        Args:
            threshold: Consensus threshold
            
        Returns:
            Consensus information
        """
        if self.consensus_predictions_ is None:
            return pd.DataFrame()
        
        consensus_df = self.consensus_predictions_.copy()
        
        if threshold is not None:
            consensus_df["high_consensus"] = (
                consensus_df["n_agreeing"] / len(self.specialists) >= threshold
            ).astype(int)
        
        return consensus_df
    
    def get_manager_summary(self) -> Dict[str, Any]:
        """
        Get summary of specialist manager.
        
        Returns:
            Manager summary
        """
        summary = {
            "n_specialists": len(self.specialists),
            "specialist_names": [s.name for s in self.specialists],
            "consensus_threshold": self.consensus_threshold,
            "surprise_aggregation": self.surprise_aggregation,
            "has_consensus_predictions": self.consensus_predictions_ is not None,
            "has_aggregated_surprises": self.aggregated_surprises_ is not None
        }
        
        # Add specialist summaries
        specialist_summaries = {}
        for specialist in self.specialists:
            specialist_summaries[specialist.name] = specialist.get_summary()
        
        summary["specialist_summaries"] = specialist_summaries
        
        return summary

# Convenience functions
def create_causal_specialists(
    causal_graph: Dict[str, List[str]],
    model_type: str = "linear",
    **kwargs
) -> CausalSpecialistManager:
    """
    Create causal specialists from causal graph.
    
    Args:
        causal_graph: Causal graph
        model_type: Model type for specialists
        **kwargs: Additional parameters
        
    Returns:
        CausalSpecialistManager instance
    """
    manager_keys = {"consensus_threshold", "surprise_aggregation", "verbose"}
    manager_kwargs = {k: v for k, v in kwargs.items() if k in manager_keys}
    specialist_kwargs = {k: v for k, v in kwargs.items() if k not in manager_keys}

    specialists = []
    
    for child, parents in causal_graph.items():
        for parent in parents:
            specialist_name = f"{parent}_to_{child}"
            specialist = CausalSpecialist(
                name=specialist_name,
                causal_parent=parent,
                causal_child=child,
                model_type=model_type,
                **specialist_kwargs
            )
            specialists.append(specialist)
    
    return CausalSpecialistManager(specialists, **manager_kwargs)

def quick_specialist_training(
    X: pd.DataFrame,
    y_dict: Dict[str, pd.Series],
    causal_graph: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> CausalSpecialistManager:
    """
    Quick specialist training.
    
    Args:
        X: Feature matrix
        y_dict: Target dictionary
        causal_graph: Causal graph
        **kwargs: Additional parameters
        
    Returns:
        Trained CausalSpecialistManager
    """
    if causal_graph:
        manager = create_causal_specialists(causal_graph, **kwargs)
    else:
        manager = CausalSpecialistManager(**kwargs)
    
    manager.fit_all_specialists(X, y_dict)
    return manager


@dataclass
class LiquidationSpecialist(CausalSpecialist):
    """
    Specialist focused on Liquidation Squeezes.
    
    Logic:
    - Identifies proximity to Liquidation Proxy Levels (Pivot + ATR).
    - Checks for Volume Stress (RVS > Threshold).
    - Triggers 'Squeeze' signal if Price dips into Proxy Zone with High Stress.
    """
    
    def __init__(
        self,
        name: str = "Liquidation_Specialist",
        rvs_threshold: float = 2.0,
        **kwargs
    ):
        super().__init__(name, causal_parent="liquidation_risk_long", causal_child="returns", **kwargs)
        self.rvs_threshold = rvs_threshold
        self.confidence_scores_ = None # Initialize 
        
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Fit method (Standard Linear Model on composite feature).
        
        We primarily use the 'liquidation_risk_long' composite feature created in feature generation,
        but we can also look at individual components if available.
        """
        if self.verbose:
            tprint_info(f"🌊 Training Liquidation Specialist: {self.name}")
        
        # We use the standard fit from parent, but ensure we target the composite risk score
        # passing 'liquidation_risk_long' as the parent feature.
        return super().fit(X, y, sample_weights)

    def detect_squeeze(self, X: pd.DataFrame) -> pd.Series:
        """
        Detect Squeeze Events based on Hard Logic rules.
        
        Rule:
        - Price < Long_Proxy (implied by positive dist_to_long_proxy if normalized correctly, 
          OR check 'in_long_liq_zone' flag if available).
        - RVS > Threshold.
        """
        # We can reconstruct logic from raw features or use the composite
        # Composite 'liquidation_risk_long' = in_zone * high_stress
        if 'liquidation_risk_long' in X.columns:
            # If feature gen did the heavy lifting
            return (X['liquidation_risk_long'] > 0).astype(int)
            
        # Fallback reconstruction
        squeeze_signal = pd.Series(0, index=X.index)
        
        if 'dist_to_long_proxy' in X.columns and 'relative_volume_stress' in X.columns:
            # dist_to_long_proxy = (Price - Proxy). 
            # If Price < Proxy, dist is negative.
            # Wait, feature gen said: "For Long Squeeze (price drops to proxy): Distance is positive if Price > Proxy"
            # mtf_feature_generation.py: dist_to_long_proxy = (close - proxy_long) / atr
            # So if Price < Proxy, dist is NEGATIVE.
            
            # BUT, the user prompt said: "Price < Long_Proxy_Level".
            # So we look for dist_to_long_proxy < 0.
            
            in_zone = X['dist_to_long_proxy'] < 0
            high_stress = X['relative_volume_stress'] > self.rvs_threshold
            
            squeeze_signal = (in_zone & high_stress).astype(int)
            
        return squeeze_signal

    def predict(self, X: pd.DataFrame, return_confidence: bool = True) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Predict returns, boosting them if a Squeeze is detected.
        """
        # Base prediction from linear model
        base_pred = super().predict(X, return_confidence=False)
        
        # Add Squeeze Boost
        squeeze = self.detect_squeeze(X)
        
        if return_confidence:
            confidence = pd.Series(0.5, index=X.index)
            # Boost confidence where squeeze is detected
            confidence[squeeze == 1] = 0.9
            # Use base confidence for others?
            if self.confidence_scores_ is not None and not self.confidence_scores_.empty:
                 # Align base confidence
                 base_conf = self.confidence_scores_.reindex(X.index).fillna(0.5)
                 confidence = np.maximum(confidence, base_conf)
                 
            return base_pred, confidence
        
        return base_pred

