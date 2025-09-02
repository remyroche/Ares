"""
Enhanced Scenario-Based Predictor for Tactician

Implements advanced probabilistic scenario analysis with:
- All step7 technical indicators
- 15-minute look-ahead period
- Fractal scenario definitions (linear progression)
- Full step17 optimization for all parameters
- Complete migration from existing system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import logging
import talib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

# Simple error handling decorator
def handle_errors(func):
    """Simple error handling decorator."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


@dataclass
class ScenarioConfig:
    """Configuration for scenario analysis."""
    profit_zones: List[float]
    risk_zones: List[float]
    time_limit_minutes: int
    decision_thresholds: Dict[str, float]
    model_config: Dict[str, Any]


class EnhancedScenarioBasedPredictor:
    """
    Enhanced scenario-based predictor with fractal scenarios and comprehensive technical indicators.

    Fractal Scenarios (Linear Progression):
    - Profit Zones: 0.25%, 0.5%, 0.75%, 1.0%, 1.25%, 1.5%, 1.75%, 2.0%
    - Risk Zones: -0.25%, -0.5%, -0.75%, -1.0%, -1.25%, -1.5%, -1.75%, -2.0%
    - Neutral: No scenario triggered within 15 minutes
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the enhanced scenario-based predictor."""
        self.config = config
        self.logger = logger

        # Load step17 optimization parameters
        step17_config = config.get("step17_optimization", {})
        scenario_config = step17_config.get("enhanced_scenario_analysis", {})

        # Fractal scenario definitions (configurable for step17)
        self.scenarios = self._create_fractal_scenarios(scenario_config)

        # Time limit for scenario evaluation (15 minutes)
        self.time_limit_minutes = scenario_config.get("time_limit_minutes", 15)

        # Model configuration (configurable for step17)
        self.model_config = {
            "n_estimators": scenario_config.get("n_estimators", 200),
            "learning_rate": scenario_config.get("learning_rate", 0.05),
            "max_depth": scenario_config.get("max_depth", 8),
            "num_leaves": scenario_config.get("num_leaves", 63),
            "subsample": scenario_config.get("subsample", 0.8),
            "colsample_bytree": scenario_config.get("colsample_bytree", 0.8),
            "random_state": scenario_config.get("random_state", 42),
            "verbose": -1
        }

        # Enhanced decision thresholds (configurable for step17)
        self.decision_thresholds = {
            "profit_zone_combined": scenario_config.get("profit_zone_combined_threshold", 0.6),
            "risk_zone_combined": scenario_config.get("risk_zone_combined_threshold", 0.2),
            "exit_risk_threshold": scenario_config.get("exit_risk_threshold", 0.5),
            "neutral_threshold": scenario_config.get("neutral_threshold", 0.3),
        }

        # Model and state variables
        self.model: Optional[lgb.LGBMClassifier] = None
        self.is_trained: bool = False
        self.feature_importance: Dict[str, float] = {}
        self.model_performance: Dict[str, float] = {}

    def _create_fractal_scenarios(self, scenario_config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Create fractal scenario definitions with linear progression."""
        scenarios = {}
        scenario_id = 0

        # Profit zones (positive scenarios)
        profit_zones = scenario_config.get("profit_zones", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        for zone in profit_zones:
            scenarios[scenario_id] = {
                "type": "profit",
                "threshold": zone / 100.0,  # Convert percentage to decimal
                "weight": 1.0 + (scenario_id * 0.1),  # Increasing weight for higher zones
                "description": f"Profit zone {zone}%"
            }
            scenario_id += 1

        # Risk zones (negative scenarios)
        risk_zones = scenario_config.get("risk_zones", [-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0])
        for zone in risk_zones:
            scenarios[scenario_id] = {
                "type": "risk",
                "threshold": zone / 100.0,  # Convert percentage to decimal
                "weight": 1.0 + (abs(scenario_id) * 0.1),  # Increasing weight for higher risk
                "description": f"Risk zone {zone}%"
            }
            scenario_id += 1

        # Neutral scenario (no significant movement)
        scenarios[scenario_id] = {
            "type": "neutral",
            "threshold": 0.0,
            "weight": 1.0,
            "description": "Neutral - no significant movement"
        }

        self.logger.info(f"Created {len(scenarios)} fractal scenarios")
        return scenarios

    async def initialize(self) -> bool:
        """Initialize the predictor asynchronously."""
        try:
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Configuration validation failed")
                return False

            # Initialize model
            self.model = lgb.LGBMClassifier(**self.model_config)
            
            self.logger.info("✅ Enhanced Scenario-Based Predictor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Enhanced Scenario-Based Predictor initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate the configuration parameters."""
        try:
            # Check required parameters
            if not self.scenarios:
                self.logger.error("No scenarios defined")
                return False

            if self.time_limit_minutes <= 0:
                self.logger.error("Invalid time limit")
                return False

            if not self.model_config:
                self.logger.error("No model configuration")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    @handle_errors
    def extract_comprehensive_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive technical indicators for step7 analysis."""
        try:
            features = []
            
            # Price-based features
            features.append(data['close'].pct_change().values)
            features.append(data['high'].pct_change().values)
            features.append(data['low'].pct_change().values)
            features.append(data['volume'].pct_change().values)

            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                ma = data['close'].rolling(period).mean()
                features.append((data['close'] / ma - 1).values)
                features.append((data['volume'] / data['volume'].rolling(period).mean() - 1).values)

            # RSI
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            features.append(rsi / 100.0)  # Normalize to 0-1

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
            features.append(macd / data['close'].values)  # Normalize
            features.append(macd_signal / data['close'].values)
            features.append(macd_hist / data['close'].values)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
            features.append((data['close'].values - bb_middle) / (bb_upper - bb_lower))

            # Stochastic
            slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
            features.append(slowk / 100.0)
            features.append(slowd / 100.0)

            # ATR (Average True Range)
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values)
            features.append(atr / data['close'].values)

            # Williams %R
            willr = talib.WILLR(data['high'].values, data['low'].values, data['close'].values)
            features.append(willr / 100.0)

            # CCI (Commodity Channel Index)
            cci = talib.CCI(data['high'].values, data['low'].values, data['close'].values)
            features.append(cci / 100.0)

            # Convert to numpy array and handle NaN values
            feature_array = np.column_stack(features)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

            self.logger.info(f"Extracted {feature_array.shape[1]} features from {len(data)} data points")
            return feature_array

        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            return np.array([])

    def prepare_scenario_targets(self, data: pd.DataFrame, lookahead_minutes: int = 15) -> np.ndarray:
        """Prepare scenario targets based on future price movements."""
        try:
            # Calculate future returns at the specified lookahead
            future_returns = data['close'].shift(-lookahead_minutes).pct_change(lookahead_minutes)
            
            # Create scenario labels
            targets = []
            for return_val in future_returns:
                if pd.isna(return_val):
                    targets.append(len(self.scenarios) - 1)  # Default to neutral
                else:
                    targets.append(self._determine_first_scenario(return_val))
            
            return np.array(targets)

        except Exception as e:
            self.logger.error(f"❌ Scenario target preparation failed: {e}")
            return np.array([0.5] * 150)

    def _determine_first_scenario(self, return_val: float) -> int:
        """Determine which scenario is triggered first by the return value."""
        try:
            for scenario_id, scenario in self.scenarios.items():
                if scenario['type'] == 'neutral':
                    continue
                
                threshold = scenario['threshold']
                if scenario['type'] == 'profit' and return_val >= threshold:
                    return scenario_id
                elif scenario['type'] == 'risk' and return_val <= threshold:
                    return scenario_id
            
            # Default to neutral if no scenario triggered
            return len(self.scenarios) - 1

        except Exception as e:
            self.logger.error(f"❌ Scenario determination failed: {e}")
            return len(self.scenarios) - 1

    def _scenario_triggered(self, return_val: float, scenario_id: int) -> bool:
        """Check if a specific scenario is triggered."""
        try:
            if scenario_id >= len(self.scenarios):
                return False
            
            scenario = self.scenarios[scenario_id]
            threshold = scenario['threshold']
            
            if scenario['type'] == 'profit':
                return return_val >= threshold
            elif scenario['type'] == 'risk':
                return return_val <= threshold
            else:  # neutral
                return abs(return_val) < 0.001  # Very small movement
                
        except Exception as e:
            self.logger.error(f"❌ Scenario trigger check failed: {e}")
            return False

    @handle_errors
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """Train the scenario prediction model."""
        try:
            self.logger.info("🔄 Starting model training...")
            
            # Extract features
            X = self.extract_comprehensive_features(training_data)
            if X.size == 0:
                self.logger.error("No features extracted for training")
                return False

            # Prepare targets
            y = self.prepare_scenario_targets(training_data, self.time_limit_minutes)
            if len(y) == 0:
                self.logger.error("No targets prepared for training")
                return False

            # Align data lengths
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            log_loss_score = log_loss(y_test, y_pred_proba)
            
            # Store performance metrics
            self.model_performance = {
                "accuracy": accuracy,
                "log_loss": log_loss_score,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_scenarios": len(self.scenarios)
            }
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    [f"feature_{i}" for i in range(X.shape[1])],
                    self.model.feature_importances_
                ))
            
            self.is_trained = True
            self.logger.info(f"✅ Model training completed. Accuracy: {accuracy:.4f}, Log Loss: {log_loss_score:.4f}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            return False

    async def predict_scenarios(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict scenario probabilities for the given data."""
        try:
            if not self.is_trained or self.model is None:
                self.logger.error("Model not trained")
                return {}

            # Extract features
            X = self.extract_comprehensive_features(data)
            if X.size == 0:
                return {}

            # Get predictions
            scenario_probs = self.model.predict_proba(X)
            scenario_predictions = self.model.predict(X)
            
            # Calculate combined probabilities
            profit_prob = np.sum(scenario_probs[:, [i for i, s in self.scenarios.items() if s['type'] == 'profit']], axis=1)
            risk_prob = np.sum(scenario_probs[:, [i for i, s in self.scenarios.items() if s['type'] == 'risk']], axis=1)
            neutral_prob = np.sum(scenario_probs[:, [i for i, s in self.scenarios.items() if s['type'] == 'neutral']], axis=1)
            
            # Make decisions based on thresholds
            decisions = []
            for i in range(len(data)):
                if profit_prob[i] >= self.decision_thresholds['profit_zone_combined']:
                    decisions.append('BUY')
                elif risk_prob[i] >= self.decision_thresholds['risk_zone_combined']:
                    decisions.append('SELL')
                elif neutral_prob[i] >= self.decision_thresholds['neutral_threshold']:
                    decisions.append('HOLD')
                else:
                    decisions.append('WAIT')
            
            return {
                "scenario_probabilities": scenario_probs.tolist(),
                "scenario_predictions": scenario_predictions.tolist(),
                "combined_probabilities": {
                    "profit": profit_prob.tolist(),
                    "risk": risk_prob.tolist(),
                    "neutral": neutral_prob.tolist()
                },
                "decisions": decisions,
                "confidence_scores": np.max(scenario_probs, axis=1).tolist()
            }

        except Exception as e:
            self.logger.error(f"❌ Scenario prediction failed: {e}")
            return {}

    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        return self.model_performance.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()

    def get_scenarios(self) -> Dict[int, Dict[str, Any]]:
        """Get scenario definitions."""
        return self.scenarios.copy()

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update decision thresholds."""
        self.decision_thresholds.update(new_thresholds)
        self.logger.info(f"Updated thresholds: {new_thresholds}")

    def reset_model(self) -> None:
        """Reset the trained model."""
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
        self.model_performance = {}
        self.logger.info("Model reset successfully")