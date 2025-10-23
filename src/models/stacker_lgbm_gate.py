"""
Regime-aware gating stacker that blends Analyst and Tactician experts.

This module factors the gating functionality that was previously embedded in
``stacker_lgbm_calibrated`` into a dedicated component so the original
LightGBM-based stacker can remain available for other consumers (e.g. the
Tactician). The gate consumes regime/context features alongside the analysts'
and tacticians' out-of-fold predictions, produces softmax-normalised expert
weights subject to an entropy penalty, and calibrates the blended probability
output for inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

EPS = 1e-9

@dataclass
class RegimeGatingConfig:
    """Configuration specific to the regime-aware gating head."""

    learning_rate: float = 0.05
    epochs: int = 400
    entropy_penalty: float = 0.01
    monotonic_features: Tuple[str, ...] = ("volatility_level", "trend_score")
    weight_clip: float = 10.0
    random_state: int = 42

@dataclass
class StackerLGBMGateConfig:
    """Configuration for the gated stacker and calibration layers."""

    # Calibration parameters
    calibration_method: str = "isotonic"  # or "sigmoid"

    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1

    # Regime gating configuration
    gating: RegimeGatingConfig = field(default_factory=RegimeGatingConfig)

class RegimeGatingHead:
    """Simple neural gating head optimised via gradient descent."""

    def __init__(
        self,
        feature_names: List[str],
        expert_names: Tuple[str, ...],
        config: RegimeGatingConfig,
    ) -> None:
        self.feature_names = feature_names
        self.expert_names = expert_names
        self.config = config
        rng = np.random.default_rng(config.random_state)
        self.weights = rng.normal(scale=0.01, size=(len(expert_names), len(feature_names)))
        self.bias = np.zeros(len(expert_names))
        self.loss_history: List[float] = []
        self.monotonic_indices = [
            feature_names.index(name)
            for name in config.monotonic_features
            if name in feature_names
        ]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @staticmethod
    def _prepare_sample_weight(
        sample_weight: Optional[np.ndarray], n_samples: int
    ) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n_samples)
        sample_weight = np.asarray(sample_weight).reshape(-1)
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Sample weight length does not match number of samples")
        return sample_weight

    def _enforce_monotonicity(self) -> None:
        if len(self.expert_names) < 2 or not self.monotonic_indices:
            return
        analyst_idx = 0  # Analyst expected first
        tactician_idx = 1 if len(self.expert_names) > 1 else 0
        for j in self.monotonic_indices:
            if self.weights[analyst_idx, j] < self.weights[tactician_idx, j]:
                midpoint = (self.weights[analyst_idx, j] + self.weights[tactician_idx, j]) / 2.0
                self.weights[analyst_idx, j] = midpoint + 1e-6
                self.weights[tactician_idx, j] = midpoint - 1e-6
        self.weights = np.clip(self.weights, -self.config.weight_clip, self.config.weight_clip)
        self.bias = np.clip(self.bias, -self.config.weight_clip, self.config.weight_clip)

    def fit(
        self,
        X: np.ndarray,
        expert_probs: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        n_samples, _ = X.shape
        sample_weight = self._prepare_sample_weight(sample_weight, n_samples)
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError("Target length does not match number of samples")

        for _ in range(self.config.epochs):
            logits = X @ self.weights.T + self.bias
            weights = self._softmax(logits)
            combined = np.clip(np.sum(weights * expert_probs, axis=1), EPS, 1 - EPS)

            base_loss = -(
                y * np.log(combined)
                + (1 - y) * np.log(1 - combined)
            )
            entropy = np.sum(weights * np.log(weights + EPS), axis=1)
            total_loss = (base_loss + self.config.entropy_penalty * entropy) * sample_weight
            loss = total_loss.sum() / np.sum(sample_weight)
            self.loss_history.append(float(loss))

            common_term = (combined - y) / (combined * (1 - combined) + EPS)
            dL_dweights = (common_term[:, None] * expert_probs)
            dL_dweights += self.config.entropy_penalty * (np.log(weights + EPS) + 1.0)
            dL_dweights *= sample_weight[:, None]

            dL_dlogits = weights * (
                dL_dweights - np.sum(weights * dL_dweights, axis=1, keepdims=True)
            )

            grad_W = (dL_dlogits.T @ X) / np.sum(sample_weight)
            grad_b = np.sum(dL_dlogits, axis=0) / np.sum(sample_weight)

            self.weights -= self.config.learning_rate * grad_W
            self.bias -= self.config.learning_rate * grad_b
            self._enforce_monotonicity()

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights.T + self.bias
        return self._softmax(logits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "feature_names": self.feature_names,
            "expert_names": list(self.expert_names),
            "loss_history": self.loss_history,
            "config": {
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "entropy_penalty": self.config.entropy_penalty,
                "monotonic_features": list(self.config.monotonic_features),
                "weight_clip": self.config.weight_clip,
                "random_state": self.config.random_state,
            },
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "RegimeGatingHead":
        config = RegimeGatingConfig(
            learning_rate=state.get("config", {}).get("learning_rate", 0.05),
            epochs=state.get("config", {}).get("epochs", 400),
            entropy_penalty=state.get("config", {}).get("entropy_penalty", 0.01),
            monotonic_features=tuple(state.get("config", {}).get("monotonic_features", [])),
            weight_clip=state.get("config", {}).get("weight_clip", 10.0),
            random_state=state.get("config", {}).get("random_state", 42),
        )
        feature_names = state.get("feature_names", [])
        expert_names = tuple(state.get("expert_names", []))
        head = cls(feature_names, expert_names, config)
        head.weights = np.asarray(state.get("weights", head.weights))
        head.bias = np.asarray(state.get("bias", head.bias))
        head.loss_history = list(state.get("loss_history", []))
        return head

class StackerLGBMGate(BaseEstimator, RegressorMixin):
    """Regime-aware gated stacker with calibration."""

    def __init__(self, config: Optional[StackerLGBMGateConfig] = None) -> None:
        self.config = config or StackerLGBMGateConfig()
        self.gating_head: Optional[RegimeGatingHead] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibration_model: Optional[Any] = None
        self.calibration_method: Optional[str] = None
        self.expert_names: List[str] = []
        self.regime_feature_names: List[str] = []
        self.base_feature_names: List[str] = []
        self.gating_feature_names: List[str] = []
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        base_predictions: Dict[str, Any],
        y: np.ndarray,
        regime_features: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "StackerLGBMGate":
        if regime_features is None:
            raise ValueError("Regime features are required for gating head training")

        y = np.asarray(y).reshape(-1)
        prob_map, _, n_samples = self._extract_expert_probabilities(base_predictions)
        if y.shape[0] != n_samples:
            raise ValueError("Target length does not match number of expert predictions")

        expert_names = self._determine_expert_order(prob_map)
        prob_matrix = np.column_stack([prob_map[name] for name in expert_names])
        features_matrix, regime_names, base_feature_names = self._build_feature_matrix_for_fit(
            regime_features,
            prob_map,
            expert_names,
            n_samples,
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_matrix)

        self.gating_head = RegimeGatingHead(
            feature_names=regime_names + base_feature_names,
            expert_names=tuple(expert_names),
            config=self.config.gating,
        )
        self.gating_head.fit(X_scaled, prob_matrix, y, sample_weight=sample_weight)

        weights = self.gating_head.predict_weights(X_scaled)
        combined_prob = np.clip(np.sum(weights * prob_matrix, axis=1), EPS, 1 - EPS)
        self._fit_calibration(combined_prob, y, sample_weight)

        self.expert_names = expert_names
        self.regime_feature_names = regime_names
        self.base_feature_names = base_feature_names
        self.gating_feature_names = regime_names + base_feature_names
        self.fitted = True
        return self

    def _determine_expert_order(self, prob_map: Dict[str, np.ndarray]) -> List[str]:
        preferred = ["analyst", "tactician"]
        ordered = [name for name in preferred if name in prob_map]
        ordered.extend([name for name in prob_map.keys() if name not in ordered])
        if len(ordered) < 2:
            raise ValueError("At least two experts (analyst and tactician) are required")
        return ordered

    def _extract_expert_probabilities(
        self, base_predictions: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
        prob_map: Dict[str, np.ndarray] = {}
        utility_map: Dict[str, np.ndarray] = {}
        n_samples: Optional[int] = None

        for expert_name, payload in base_predictions.items():
            if payload is None:
                continue
            probability = self._extract_probability_array(payload)
            if probability is None:
                logger.warning("⚠️ Missing probability predictions for expert %s", expert_name)
                continue
            probability = probability.reshape(-1)
            if n_samples is None:
                n_samples = probability.shape[0]
            elif probability.shape[0] != n_samples:
                raise ValueError("All expert predictions must have the same length")
            prob_map[expert_name] = probability

            utility = self._extract_utility_array(payload)
            if utility is not None:
                utility_map[expert_name] = utility.reshape(-1)

        if n_samples is None:
            raise ValueError("No expert predictions provided")
        return prob_map, utility_map, n_samples

    def _extract_probability_array(self, payload: Any) -> Optional[np.ndarray]:
        candidates: List[np.ndarray] = []
        if isinstance(payload, dict):
            for key in ("probability", "probabilities", "proba", "prediction", "predictions"):
                if key in payload and payload[key] is not None:
                    arr = np.asarray(payload[key])
                    if arr.ndim == 1:
                        candidates.append(arr)
                    elif arr.ndim == 2:
                        if arr.shape[1] == 1:
                            candidates.append(arr[:, 0])
                        elif arr.shape[1] >= 2:
                            candidates.append(arr[:, 1])
        else:
            arr = np.asarray(payload)
            if arr.ndim == 1:
                candidates.append(arr)
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                candidates.append(arr[:, 1])
        if not candidates:
            return None
        return np.asarray(candidates[0]).astype(float)

    def _extract_utility_array(self, payload: Any) -> Optional[np.ndarray]:
        if isinstance(payload, dict):
            for key in ("utility", "utilities", "expected_utility", "reward", "expected_reward"):
                if key in payload and payload[key] is not None:
                    return np.asarray(payload[key]).astype(float)
        return None

    def _build_feature_matrix_for_fit(
        self,
        regime_features: Dict[str, Any],
        prob_map: Dict[str, np.ndarray],
        expert_names: List[str],
        n_samples: int,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        regime_order = self._derive_regime_feature_order(regime_features)
        regime_matrix = [
            self._get_regime_feature_value(name, regime_features, n_samples).reshape(-1, 1)
            for name in regime_order
        ]
        base_feature_names: List[str] = []
        base_matrix: List[np.ndarray] = []

        for expert in expert_names:
            base_matrix.append(prob_map[expert].reshape(-1, 1))
            base_feature_names.append(f"{expert}_oof_probability")

        if len(expert_names) >= 2:
            left, right = expert_names[0], expert_names[1]
            base_matrix.append((prob_map[left] - prob_map[right]).reshape(-1, 1))
            base_feature_names.append(f"{left}_minus_{right}_oof")

        features_matrix = np.column_stack(regime_matrix + base_matrix)
        return features_matrix, regime_order, base_feature_names

    def _derive_regime_feature_order(self, regime_features: Dict[str, Any]) -> List[str]:
        if not regime_features:
            return []
        available = list(regime_features.keys())
        ordered: List[str] = []
        for name in self.config.gating.monotonic_features:
            if name in regime_features and name not in ordered:
                ordered.append(name)
        for candidate in ["liquidity_z", "liquidity_score"]:
            if candidate in regime_features and candidate not in ordered:
                ordered.append(candidate)
        for name in sorted(available):
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _get_regime_feature_value(
        self, name: str, regime_features: Dict[str, Any], n_samples: int
    ) -> np.ndarray:
        value: Optional[Any] = None
        if isinstance(regime_features, dict) and name in regime_features:
            value = regime_features[name]
        elif isinstance(regime_features, pd.DataFrame) and name in regime_features.columns:
            value = regime_features[name].values
        if value is None:
            logger.warning("⚠️ Missing regime feature %s; defaulting to zeros", name)
            return np.zeros(n_samples)
        arr = np.asarray(value).astype(float)
        if arr.shape[0] != n_samples:
            raise ValueError(f"Regime feature {name} length mismatch")
        return arr

    def _fit_calibration(
        self,
        combined_prob: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        method = (self.config.calibration_method or "").lower()
        self.calibration_model = None
        self.calibration_method = None
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(combined_prob, y, sample_weight=sample_weight)
            self.calibration_model = model
            self.calibration_method = "isotonic"
        elif method == "sigmoid":
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state,
            )
            model.fit(combined_prob.reshape(-1, 1), y, sample_weight=sample_weight)
            self.calibration_model = model
            self.calibration_method = "sigmoid"

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    def _build_feature_matrix_for_inference(
        self,
        regime_features: Optional[Dict[str, Any]],
        prob_map: Dict[str, np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        regime_matrix: List[np.ndarray] = []
        for name in self.regime_feature_names:
            value = self._safe_regime_feature_for_inference(name, regime_features, n_samples)
            regime_matrix.append(value.reshape(-1, 1))

        base_matrix: List[np.ndarray] = []
        for base_name in self.base_feature_names:
            if base_name.endswith("_oof_probability"):
                expert = base_name[: -len("_oof_probability")]
                base_matrix.append(prob_map[expert].reshape(-1, 1))
            elif base_name.endswith("_oof") and "_minus_" in base_name:
                core = base_name[: -len("_oof")]
                left, right = core.split("_minus_")
                base_matrix.append((prob_map[left] - prob_map[right]).reshape(-1, 1))
            else:
                raise ValueError(f"Unrecognized base feature name: {base_name}")

        if not regime_matrix and not base_matrix:
            raise ValueError("No features available for gating inference")
        return np.column_stack(regime_matrix + base_matrix)

    def _safe_regime_feature_for_inference(
        self, name: str, regime_features: Optional[Dict[str, Any]], n_samples: int
    ) -> np.ndarray:
        if regime_features is None:
            return np.zeros(n_samples)
        if isinstance(regime_features, dict) and name in regime_features:
            arr = np.asarray(regime_features[name]).astype(float)
        elif isinstance(regime_features, pd.DataFrame) and name in regime_features.columns:
            arr = np.asarray(regime_features[name]).astype(float)
        else:
            logger.warning("⚠️ Missing regime feature %s during inference; using zeros", name)
            return np.zeros(n_samples)
        if arr.shape[0] != n_samples:
            raise ValueError(f"Regime feature {name} length mismatch during inference")
        return arr

    def _apply_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        if self.calibration_model is None:
            return probabilities
        if self.calibration_method == "isotonic":
            return np.clip(self.calibration_model.predict(probabilities), 0.0, 1.0)
        if self.calibration_method == "sigmoid":
            return np.clip(
                self.calibration_model.predict_proba(probabilities.reshape(-1, 1))[:, 1],
                0.0,
                1.0,
            )
        return probabilities

    def _prepare_inputs(
        self,
        base_predictions: Dict[str, Any],
        regime_features: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        prob_map, utility_map, n_samples = self._extract_expert_probabilities(base_predictions)
        missing = [name for name in self.expert_names if name not in prob_map]
        if missing:
            raise ValueError(f"Missing predictions for experts: {missing}")
        prob_matrix = np.column_stack([prob_map[name] for name in self.expert_names])
        features_matrix = self._build_feature_matrix_for_inference(regime_features, prob_map, n_samples)
        X_scaled = self.scaler.transform(features_matrix)
        weights = self.gating_head.predict_weights(X_scaled)
        return weights, prob_matrix, utility_map

    def predict(
        self,
        base_predictions: Dict[str, Any],
        regime_features: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        weights, prob_matrix, _ = self._prepare_inputs(base_predictions, regime_features)
        combined_prob = np.clip(np.sum(weights * prob_matrix, axis=1), EPS, 1 - EPS)
        calibrated = self._apply_calibration(combined_prob)
        return calibrated

    def predict_proba(
        self,
        base_predictions: Dict[str, Any],
        regime_features: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        calibrated = self.predict(base_predictions, regime_features)
        return np.column_stack([1 - calibrated, calibrated])

    def combine_outputs(
        self,
        base_predictions: Dict[str, Any],
        regime_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        weights, prob_matrix, utility_map = self._prepare_inputs(base_predictions, regime_features)
        combined_prob = np.clip(np.sum(weights * prob_matrix, axis=1), EPS, 1 - EPS)
        calibrated_prob = self._apply_calibration(combined_prob)

        weighted_utilities: Optional[np.ndarray] = None
        if utility_map:
            utility_stack = []
            for idx, expert in enumerate(self.expert_names):
                if expert in utility_map:
                    utility_stack.append(weights[:, idx] * utility_map[expert])
                else:
                    utility_stack.append(weights[:, idx] * 0.0)
            weighted_utilities = np.sum(np.vstack(utility_stack), axis=0)

        weight_dict = {
            expert: weights[:, idx] for idx, expert in enumerate(self.expert_names)
        }
        expert_probabilities = {
            expert: prob_matrix[:, idx] for idx, expert in enumerate(self.expert_names)
        }

        return {
            "probability": calibrated_prob,
            "raw_probability": combined_prob,
            "weights": weight_dict,
            "expert_probabilities": expert_probabilities,
            "utility": weighted_utilities,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def get_gating_state(self) -> Optional[Dict[str, Any]]:
        if not self.fitted or self.gating_head is None or self.scaler is None:
            return None
        return {
            "gating_head": self.gating_head.to_dict(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "regime_feature_names": self.regime_feature_names,
            "base_feature_names": self.base_feature_names,
            "expert_names": self.expert_names,
            "gating_feature_names": self.gating_feature_names,
        }

    def load_gating_state(self, state: Dict[str, Any]) -> None:
        if not state:
            raise ValueError("Invalid gating state provided")
        self.regime_feature_names = state.get("regime_feature_names", [])
        self.base_feature_names = state.get("base_feature_names", [])
        self.expert_names = state.get("expert_names", [])
        self.gating_feature_names = state.get("gating_feature_names", [])
        gating_head_state = state.get("gating_head")
        if gating_head_state is None:
            raise ValueError("Gating head state missing")
        self.gating_head = RegimeGatingHead.from_dict(gating_head_state)
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.asarray(state.get("scaler_mean", []))
        self.scaler.scale_ = np.asarray(state.get("scaler_scale", []))
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.fitted = True

    def get_calibration_state(self) -> Optional[Dict[str, Any]]:
        if self.calibration_model is None:
            return None
        if self.calibration_method == "isotonic":
            return {
                "method": "isotonic",
                "X_": self.calibration_model.X_.tolist(),
                "y_": self.calibration_model.y_.tolist(),
            }
        if self.calibration_method == "sigmoid":
            return {
                "method": "sigmoid",
                "coef_": self.calibration_model.coef_.tolist(),
                "intercept_": self.calibration_model.intercept_.tolist(),
                "classes_": self.calibration_model.classes_.tolist(),
            }
        return None

    def load_calibration_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            self.calibration_model = None
            self.calibration_method = None
            return
        method = state.get("method")
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.X_ = np.asarray(state.get("X_", []))
            model.y_ = np.asarray(state.get("y_", []))
            self.calibration_model = model
            self.calibration_method = "isotonic"
        elif method == "sigmoid":
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state,
            )
            model.coef_ = np.asarray(state.get("coef_", [[]]))
            model.intercept_ = np.asarray(state.get("intercept_", [0.0]))
            model.classes_ = np.asarray(state.get("classes_", [0, 1]))
            self.calibration_model = model
            self.calibration_method = "sigmoid"
        else:
            raise ValueError(f"Unsupported calibration method: {method}")

    def get_gating_loss_history(self) -> List[float]:
        if self.gating_head is None:
            return []
        return list(self.gating_head.loss_history)

def create_stacker_lgbm_gate(
    config: Optional[StackerLGBMGateConfig] = None,
) -> StackerLGBMGate:
    """Factory helper for the gated stacker."""
    return StackerLGBMGate(config)
