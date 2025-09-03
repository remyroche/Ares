"""Scenario-Based Predictor for Tactician.

Streamlined implementation:
- Proper LightGBM initialization
- Configuration validation
- Minimal training and prediction interfaces
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ScenarioBasedPredictor:
    """Implements probabilistic scenario analysis for Tactician.

    Scenarios:
    - 0: Profit Zone 1 (Small Profit): +0.5% before -0.5%
    - 1: Profit Zone 2 (Medium Profit): +1% before -0.5%
    - 2: Profit Zone 3 (Large Profit): +1.5% before -0.5%
    - 3: Risk Zone 1 (Small Loss): -0.5% before +0.5%
    - 4: Risk Zone 2 (Medium Loss): -1% before +0.5%
    - 5: Neutral: No scenario triggered within time limit
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.logger = logger

        step17_config = config.get("step17_optimization", {})
        scenario_config = step17_config.get("scenario_analysis", {})

        self.scenarios: Dict[int, Dict[str, Any]] = {
            0: {
                "name": "Profit Zone 1 (Small Profit)",
                "profit_target": float(
                    scenario_config.get("profit_zone_1_target", 0.005)
                ),
                "stop_loss": float(
                    scenario_config.get("profit_zone_1_stop_loss", -0.005)
                ),
            },
            1: {
                "name": "Profit Zone 2 (Medium Profit)",
                "profit_target": float(
                    scenario_config.get("profit_zone_2_target", 0.01)
                ),
                "stop_loss": float(
                    scenario_config.get("profit_zone_2_stop_loss", -0.005)
                ),
            },
            2: {
                "name": "Profit Zone 3 (Large Profit)",
                "profit_target": float(
                    scenario_config.get("profit_zone_3_target", 0.015)
                ),
                "stop_loss": float(
                    scenario_config.get("profit_zone_3_stop_loss", -0.005)
                ),
            },
            3: {
                "name": "Risk Zone 1 (Small Loss)",
                "profit_target": float(
                    scenario_config.get("risk_zone_1_target", 0.005)
                ),
                "stop_loss": float(
                    scenario_config.get("risk_zone_1_stop_loss", -0.005)
                ),
            },
            4: {
                "name": "Risk Zone 2 (Medium Loss)",
                "profit_target": float(scenario_config.get("risk_zone_2_target", 0.01)),
                "stop_loss": float(
                    scenario_config.get("risk_zone_2_stop_loss", -0.005)
                ),
            },
            5: {
                "name": "Neutral",
                "profit_target": float(scenario_config.get("neutral_target", 0.0)),
                "stop_loss": float(scenario_config.get("neutral_stop_loss", 0.0)),
            },
        }

        self.time_limit_minutes: int = int(
            scenario_config.get("time_limit_minutes", 30)
        )

        self.model_config: Dict[str, Any] = {
            "n_estimators": int(scenario_config.get("n_estimators", 100)),
            "learning_rate": float(scenario_config.get("learning_rate", 0.1)),
            "max_depth": int(scenario_config.get("max_depth", 6)),
            "num_leaves": int(scenario_config.get("num_leaves", 31)),
            "subsample": float(scenario_config.get("subsample", 0.8)),
            "colsample_bytree": float(scenario_config.get("colsample_bytree", 0.8)),
            "random_state": int(scenario_config.get("random_state", 42)),
            "verbose": -1,
        }

        self.decision_thresholds: Dict[str, float] = {
            "profit_zone_combined": float(
                scenario_config.get("profit_zone_combined_threshold", 0.6)
            ),
            "risk_zone_combined": float(
                scenario_config.get("risk_zone_combined_threshold", 0.2)
            ),
            "exit_risk_threshold": float(
                scenario_config.get("exit_risk_threshold", 0.5)
            ),
            "neutral_threshold": float(scenario_config.get("neutral_threshold", 0.3)),
            "confidence_threshold": float(
                scenario_config.get("confidence_threshold", 0.7)
            ),
        }

        self.feature_config: Dict[str, int] = {
            "lookback_periods": int(scenario_config.get("lookback_periods", 20)),
            "volatility_window": int(scenario_config.get("volatility_window", 20)),
            "rsi_period": int(scenario_config.get("rsi_period", 14)),
            "ma_short_period": int(scenario_config.get("ma_short_period", 5)),
            "ma_long_period": int(scenario_config.get("ma_long_period", 20)),
            "volume_ma_period": int(scenario_config.get("volume_ma_period", 10)),
        }

        self.model: Optional[lgb.LGBMClassifier] = None
        self.is_trained: bool = False
        self.last_training_time: Optional[datetime] = None
        self.feature_importance: Dict[str, float] = {}
        self.model_performance: Dict[str, float] = {}

    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Scenario-Based Predictor...")
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for scenario predictor")
                return False
            self.model = lgb.LGBMClassifier(**self.model_config)
            self.logger.info("Scenario-Based Predictor initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Scenario-Based Predictor initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        try:
            for scenario_id, scenario in self.scenarios.items():
                if scenario_id != 5 and float(scenario["profit_target"]) <= 0:
                    self.logger.error(
                        f"Invalid profit target for scenario {scenario_id}"
                    )
                    return False
                if scenario_id != 5 and float(scenario["stop_loss"]) >= 0:
                    self.logger.error(f"Invalid stop loss for scenario {scenario_id}")
                    return False

            if self.time_limit_minutes <= 0:
                self.logger.error("Invalid time limit")
                return False

            for name, threshold in self.decision_thresholds.items():
                t = float(threshold)
                if t < 0.0 or t > 1.0:
                    self.logger.error(f"Invalid threshold for {name}")
                    return False

            for name, value in self.feature_config.items():
                if int(value) <= 0:
                    self.logger.error(f"Invalid feature parameter for {name}")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def prepare_scenario_targets(
        self,
        X: np.ndarray,
        market_data: pd.DataFrame,
        base_price_column: str = "close",
    ) -> np.ndarray:
        try:
            if len(X) != len(market_data):
                raise ValueError("Feature array and market data must have same length")
            prices = market_data[base_price_column].values
            labels = [
                self._determine_first_scenario(prices[i:], i, self.time_limit_minutes)
                for i in range(len(X))
            ]
            return np.array(labels)
        except Exception as e:
            self.logger.error(f"Scenario labeling failed: {e}")
            return np.full(len(X), 5)

    def _determine_first_scenario(
        self,
        future_prices: np.ndarray,
        current_index: int,
        time_limit: int,
    ) -> int:
        try:
            if len(future_prices) < 2:
                return 5
            current_price = float(future_prices[0])
            look_ahead = future_prices[1 : min(len(future_prices), time_limit + 1)]
            for scenario_id in [0, 1, 2, 3, 4]:
                if self._scenario_triggered(
                    look_ahead, current_price, self.scenarios[scenario_id]
                ):
                    return scenario_id
            return 5
        except Exception as e:
            self.logger.error(f"Scenario determination failed: {e}")
            return 5

    def _scenario_triggered(
        self,
        prices: np.ndarray,
        current_price: float,
        scenario: Dict[str, Any],
    ) -> bool:
        try:
            profit_target = float(scenario["profit_target"])
            stop_loss = float(scenario["stop_loss"])
            changes = (prices - current_price) / max(current_price, 1e-8)
            for change in changes:
                if profit_target > 0:
                    if change >= profit_target:
                        return True
                    if change <= stop_loss:
                        return False
                else:
                    if change <= stop_loss:
                        return True
                    if change >= abs(profit_target):
                        return False
            return False
        except Exception as e:
            self.logger.error(f"Scenario trigger check failed: {e}")
            return False

    async def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        market_data: Optional[pd.DataFrame] = None,
    ) -> bool:
        try:
            self.logger.info("Training scenario prediction model...")
            if market_data is not None and len(y_train) == len(X_train):
                y_train = self.prepare_scenario_targets(X_train, market_data)

            if X_val is None or y_val is None:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            else:
                X_train_split, y_train_split = X_train, y_train

            if self.model is None:
                raise RuntimeError("Model not initialized")

            self.model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            self.feature_importance = dict(
                zip(
                    [f"feature_{i}" for i in range(X_train.shape[1])],
                    self.model.feature_importances_,
                )
            )

            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)
            self.model_performance = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "log_loss": float(log_loss(y_val, y_proba)),
                "n_samples": float(len(X_train)),
                "n_features": float(X_train.shape[1]),
            }

            self.is_trained = True
            self.last_training_time = datetime.now()
            self.logger.info(
                f"Model trained successfully. Accuracy: {self.model_performance['accuracy']:.3f}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False

    async def predict_scenarios(
        self,
        X: np.ndarray,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("Model not trained, using fallback predictions")
                return self._generate_fallback_predictions(X)

            probabilities = self.model.predict_proba(X)
            predicted_scenario = int(self.model.predict(X)[0])
            scenario_analysis = self._analyze_scenario_probabilities(probabilities[0])
            confidence = self._calculate_confidence(probabilities[0])

            return {
                "probabilities": dict(
                    zip(range(len(probabilities[0])), probabilities[0])
                ),
                "predicted_scenario": predicted_scenario,
                "scenario_name": self.scenarios[predicted_scenario]["name"],
                "confidence": float(confidence),
                "scenario_analysis": scenario_analysis,
                "metadata": {
                    "model_type": "scenario_based",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": self.is_trained,
                    "last_training_time": (
                        self.last_training_time.isoformat()
                        if self.last_training_time
                        else None
                    ),
                },
            }
        except Exception as e:
            self.logger.error(f"Scenario prediction failed: {e}")
            return self._generate_fallback_predictions(X)

    def _analyze_scenario_probabilities(
        self, probabilities: np.ndarray
    ) -> Dict[str, Any]:
        try:
            profit_zone_prob = float(sum(probabilities[i] for i in [0, 1, 2]))
            risk_zone_prob = float(sum(probabilities[i] for i in [3, 4]))
            neutral_prob = float(probabilities[5])

            if profit_zone_prob > risk_zone_prob and profit_zone_prob > neutral_prob:
                dominant_zone = "profit"
            elif risk_zone_prob > profit_zone_prob and risk_zone_prob > neutral_prob:
                dominant_zone = "risk"
            else:
                dominant_zone = "neutral"

            risk_reward_ratio = float(profit_zone_prob / (risk_zone_prob + 1e-8))

            return {
                "profit_zone_probability": profit_zone_prob,
                "risk_zone_probability": risk_zone_prob,
                "neutral_probability": neutral_prob,
                "dominant_zone": dominant_zone,
                "risk_reward_ratio": risk_reward_ratio,
                "profit_risk_difference": float(profit_zone_prob - risk_zone_prob),
            }
        except Exception as e:
            self.logger.error(f"Scenario analysis failed: {e}")
            return {
                "profit_zone_probability": 0.0,
                "risk_zone_probability": 0.0,
                "neutral_probability": 1.0,
                "dominant_zone": "neutral",
                "risk_reward_ratio": 0.0,
                "profit_risk_difference": 0.0,
            }

    def _calculate_confidence(self, probabilities: np.ndarray) -> float:
        try:
            entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-8)))
            max_entropy = float(np.log(len(probabilities)))
            confidence = 1.0 - (entropy / max_entropy)
            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _generate_fallback_predictions(self, X: np.ndarray) -> Dict[str, Any]:
        try:
            n_scenarios = len(self.scenarios)
            base_prob = 1.0 / n_scenarios
            probabilities = [base_prob * 0.8] * (n_scenarios - 1) + [base_prob * 1.4]
            return {
                "probabilities": dict(zip(range(n_scenarios), probabilities)),
                "predicted_scenario": 5,
                "scenario_name": self.scenarios[5]["name"],
                "confidence": 0.3,
                "scenario_analysis": {
                    "profit_zone_probability": base_prob * 2.4,
                    "risk_zone_probability": base_prob * 1.6,
                    "neutral_probability": base_prob * 1.4,
                    "dominant_zone": "neutral",
                    "risk_reward_ratio": 1.5,
                    "profit_risk_difference": base_prob * 0.8,
                },
                "metadata": {
                    "model_type": "scenario_based_fallback",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": False,
                    "last_training_time": None,
                },
            }
        except Exception as e:
            self.logger.error(f"Fallback prediction generation failed: {e}")
            return {
                "probabilities": {i: 1.0 / 6 for i in range(6)},
                "predicted_scenario": 5,
                "scenario_name": "Neutral",
                "confidence": 0.0,
                "scenario_analysis": {
                    "profit_zone_probability": 0.5,
                    "risk_zone_probability": 0.33,
                    "neutral_probability": 0.17,
                    "dominant_zone": "neutral",
                    "risk_reward_ratio": 1.0,
                    "profit_risk_difference": 0.0,
                },
                "metadata": {
                    "model_type": "scenario_based_error",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": False,
                    "last_training_time": None,
                },
            }

    def extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        try:
            if len(market_data) < self.feature_config["lookback_periods"]:
                return np.array([0.5] * 15)
            close_prices = market_data["close"].values
            high_prices = market_data["high"].values
            low_prices = market_data["low"].values
            volumes = market_data["volume"].values

            current_price = close_prices[-1]
            returns = np.diff(close_prices) / close_prices[:-1]
            price_momentum_5 = (current_price - close_prices[-5]) / close_prices[-5]
            price_momentum_10 = (current_price - close_prices[-10]) / close_prices[-10]
            price_momentum_20 = (current_price - close_prices[-20]) / close_prices[-20]
            volatility_5 = np.std(returns[-5:])
            volatility_10 = np.std(returns[-10:])
            volatility_20 = np.std(returns[-20:])
            volume_trend = (
                (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
            )
            volume_ma_ratio = volumes[-1] / max(
                1e-9, np.mean(volumes[-self.feature_config["volume_ma_period"] :])
            )

            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            rsi_period = self.feature_config["rsi_period"]
            avg_gain = np.mean(gains[-rsi_period:]) if len(gains) >= rsi_period else 0
            avg_loss = np.mean(losses[-rsi_period:]) if len(losses) >= rsi_period else 0
            rs = avg_gain / max(avg_loss, 1e-9)
            rsi = 100 - (100 / (1 + rs))

            ma_short = np.mean(close_prices[-self.feature_config["ma_short_period"] :])
            ma_long = np.mean(close_prices[-self.feature_config["ma_long_period"] :])
            ma_ratio = ma_short / max(ma_long, 1e-9)

            price_range = (high_prices[-1] - low_prices[-1]) / max(current_price, 1e-9)
            upper_shadow = (high_prices[-1] - current_price) / max(current_price, 1e-9)
            lower_shadow = (current_price - low_prices[-1]) / max(current_price, 1e-9)
            latest_return = (current_price - close_prices[-2]) / max(
                close_prices[-2], 1e-9
            )

            features = [
                price_momentum_5,
                price_momentum_10,
                price_momentum_20,
                volatility_5,
                volatility_10,
                volatility_20,
                volume_trend,
                volume_ma_ratio,
                rsi / 100.0,
                ma_ratio,
                price_range,
                upper_shadow,
                lower_shadow,
                latest_return,
            ]
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.array([0.5] * 15)

    def get_configuration_summary(self) -> Dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "time_limit_minutes": self.time_limit_minutes,
            "model_config": self.model_config,
            "decision_thresholds": self.decision_thresholds,
            "feature_config": self.feature_config,
            "is_trained": self.is_trained,
            "model_performance": self.model_performance,
            "feature_importance": self.feature_importance,
        }
