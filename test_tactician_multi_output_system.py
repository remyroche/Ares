#!/usr/bin/env python3
"""
Test script for the new Tactician multi-output prediction system.
This tests the simplified approach where Tactician generates its own predictions
using 50% and 25% barriers on shorter timeframes.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Mock imports for testing
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_mock_market_data(n_samples: int = 100) -> pd.DataFrame:
    """Create mock market data for testing."""
    np.random.seed(42)

    # Generate realistic price data
    base_price = 50000
    returns = np.random.normal(0, 0.01, n_samples)  # 1% daily volatility
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_samples)
    }

    return pd.DataFrame(data)

def create_mock_analyst_barriers() -> Dict[str, float]:
    """Create mock Analyst barriers."""
    return {
        "upper_barrier": 0.02,  # 2% profit target
        "lower_barrier": -0.01  # 1% stop loss
    }

class MockMLTacticsManager:
    """Mock ML Tactics Manager for testing."""

    def __init__(self):
        self.barrier_config = {
            "fifty_percent": {
                "profit_target_multiplier": 0.5,
                "stop_loss_multiplier": 0.5,
                "timeframe": "1m"
            },
            "twenty_five_percent": {
                "profit_target_multiplier": 0.25,
                "stop_loss_multiplier": 0.25,
                "timeframe": "1m"
            }
        }

        self.green_light_thresholds = {
            "fifty_percent": 0.75,
            "twenty_five_percent": 0.8,
            "combined_threshold": 0.7
        }

        self.exit_thresholds = {
            "fifty_percent": 0.4,
            "twenty_five_percent": 0.35,
            "combined_exit_threshold": 0.45
        }

    def _calculate_tactician_barriers(self, analyst_barriers: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate Tactician barriers as 50% and 25% of Analyst barriers."""
        analyst_upper = analyst_barriers.get("upper_barrier", 0.02)
        analyst_lower = analyst_barriers.get("lower_barrier", -0.01)

        tactician_barriers = {}

        # Calculate 50% barriers
        tactician_barriers["fifty_percent"] = {
            "upper_barrier": analyst_upper * self.barrier_config["fifty_percent"]["profit_target_multiplier"],
            "lower_barrier": analyst_lower * self.barrier_config["fifty_percent"]["stop_loss_multiplier"],
            "timeframe": self.barrier_config["fifty_percent"]["timeframe"]
        }

        # Calculate 25% barriers
        tactician_barriers["twenty_five_percent"] = {
            "upper_barrier": analyst_upper * self.barrier_config["twenty_five_percent"]["profit_target_multiplier"],
            "lower_barrier": analyst_lower * self.barrier_config["twenty_five_percent"]["stop_loss_multiplier"],
            "timeframe": self.barrier_config["twenty_five_percent"]["timeframe"]
        }

        return tactician_barriers

    def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features from market data for prediction."""
        if len(market_data) < 20:
            return np.array([0.5] * 10)

        close_prices = market_data['close'].values
        high_prices = market_data['high'].values
        low_prices = market_data['low'].values
        volumes = market_data['volume'].values

        features = []

        # Price momentum
        price_momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
        features.append(price_momentum)

        # Volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns[-20:])
        features.append(volatility)

        # Volume trend
        volume_trend = (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
        features.append(volume_trend)

        # Price range
        price_range = (high_prices[-1] - low_prices[-1]) / close_prices[-1]
        features.append(price_range)

        # Moving averages
        ma_short = np.mean(close_prices[-5:])
        ma_long = np.mean(close_prices[-20:])
        ma_ratio = ma_short / ma_long if ma_long > 0 else 1.0
        features.append(ma_ratio)

        # RSI-like indicator
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)

        # Additional features
        features.extend([
            close_prices[-1] / close_prices[-2] - 1,
            np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1.0,
            (high_prices[-1] - close_prices[-1]) / close_prices[-1],
            (close_prices[-1] - low_prices[-1]) / close_prices[-1]
        ])

        return np.array(features)

    def _generate_fallback_confidence(self, barrier_type: str, features: np.ndarray) -> float:
        """Generate fallback confidence score."""
        base_confidence = 0.5

        # Adjust based on price momentum
        if len(features) > 0:
            momentum = features[0]
            if abs(momentum) > 0.01:
                base_confidence += 0.2
            elif abs(momentum) > 0.005:
                base_confidence += 0.1

        # Adjust based on volatility
        if len(features) > 1:
            volatility = features[1]
            if volatility < 0.01:
                base_confidence += 0.1
            elif volatility > 0.03:
                base_confidence -= 0.1

        # Adjust based on RSI
        if len(features) > 5:
            rsi = features[5]
            if 0.3 < rsi < 0.7:
                base_confidence += 0.1
            elif rsi < 0.2 or rsi > 0.8:
                base_confidence -= 0.1

        # Adjust for barrier type
        if barrier_type == "twenty_five_percent":
            base_confidence *= 0.9

        return np.clip(base_confidence, 0.0, 1.0)

    def _determine_direction(self, features: np.ndarray) -> str:
        """Determine price direction based on features."""
        if len(features) > 0:
            momentum = features[0]
            if momentum > 0:
                return "UP"
            else:
                return "DOWN"
        else:
            return "UP"

    def _calculate_combined_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate combined confidence from all predictions."""
        confidences = []
        weights = []

        for barrier_type, prediction in predictions.items():
            if prediction and "confidence" in prediction:
                confidences.append(prediction["confidence"])
                if barrier_type == "fifty_percent":
                    weights.append(0.6)
                else:
                    weights.append(0.4)

        if not confidences:
            return 0.5

        total_weight = sum(weights)
        if total_weight > 0:
            combined_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        else:
            combined_confidence = np.mean(confidences)

        return np.clip(combined_confidence, 0.0, 1.0)

    def _evaluate_green_light_signal(self, predictions: Dict[str, Any], combined_confidence: float) -> Dict[str, Any]:
        """Evaluate green light signal based on predictions and thresholds."""
        fifty_percent_ok = False
        twenty_five_percent_ok = False

        if "fifty_percent" in predictions and predictions["fifty_percent"]:
            fifty_confidence = predictions["fifty_percent"]["confidence"]
            fifty_percent_ok = fifty_confidence >= self.green_light_thresholds["fifty_percent"]

        if "twenty_five_percent" in predictions and predictions["twenty_five_percent"]:
            twenty_five_confidence = predictions["twenty_five_percent"]["confidence"]
            twenty_five_percent_ok = twenty_five_confidence >= self.green_light_thresholds["twenty_five_percent"]

        combined_ok = combined_confidence >= self.green_light_thresholds["combined_threshold"]

        if fifty_percent_ok and twenty_five_percent_ok and combined_ok:
            signal = "GREEN_LIGHT"
            reason = "All thresholds met"
        elif combined_ok:
            signal = "YELLOW_LIGHT"
            reason = "Combined threshold met, individual thresholds partial"
        else:
            signal = "RED_LIGHT"
            reason = "Thresholds not met"

        return {
            "signal": signal,
            "reason": reason,
            "fifty_percent_ok": fifty_percent_ok,
            "twenty_five_percent_ok": twenty_five_percent_ok,
            "combined_ok": combined_ok,
            "combined_confidence": combined_confidence,
            "thresholds": self.green_light_thresholds
        }

    async def generate_multi_output_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_barriers: Dict[str, float],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate multi-output predictions for 50% and 25% barriers."""
        # Calculate Tactician barriers
        tactician_barriers = self._calculate_tactician_barriers(analyst_barriers)

        # Generate predictions for each barrier type
        predictions = {}

        for barrier_type in ["fifty_percent", "twenty_five_percent"]:
            # Extract features
            features = self._extract_features(market_data)

            # Generate confidence and direction
            confidence = self._generate_fallback_confidence(barrier_type, features)
            direction = self._determine_direction(features)

            predictions[barrier_type] = {
                "confidence": confidence,
                "direction": direction,
                "upper_barrier": tactician_barriers[barrier_type]["upper_barrier"],
                "lower_barrier": tactician_barriers[barrier_type]["lower_barrier"],
                "timeframe": tactician_barriers[barrier_type]["timeframe"],
                "barrier_type": barrier_type
            }

        # Calculate combined confidence and green light signal
        combined_confidence = self._calculate_combined_confidence(predictions)
        green_light_signal = self._evaluate_green_light_signal(predictions, combined_confidence)

        # Add metadata
        result = {
            **predictions,
            "combined_confidence": combined_confidence,
            "green_light_signal": green_light_signal,
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "generation_timestamp": datetime.now().isoformat(),
                "model_type": "tactician_multi_output",
                "barrier_config": self.barrier_config
            }
        }

        return result

async def test_multi_output_system():
    """Test the multi-output prediction system."""
    print("🧪 Testing Tactician Multi-Output Prediction System")
    print("=" * 60)

    # Create mock data
    market_data = create_mock_market_data(100)
    analyst_barriers = create_mock_analyst_barriers()

    print(f"📊 Market data: {len(market_data)} samples")
    print(f"📈 Analyst barriers: {analyst_barriers}")
    print()

    # Create mock ML tactics manager
    ml_tactics = MockMLTacticsManager()

    # Generate predictions
    predictions = await ml_tactics.generate_multi_output_predictions(
        market_data=market_data,
        analyst_barriers=analyst_barriers,
        symbol="BTCUSDT",
        timeframe="1m"
    )

    # Display results
    print("🎯 Multi-Output Predictions:")
    print("-" * 40)

    # 50% barrier prediction
    fifty_percent = predictions["fifty_percent"]
    print(f"50% Barrier Prediction:")
    print(f"  Confidence: {fifty_percent['confidence']:.4f}")
    print(f"  Direction: {fifty_percent['direction']}")
    print(f"  Upper Barrier: {fifty_percent['upper_barrier']:.4f} ({fifty_percent['upper_barrier']*100:.2f}%)")
    print(f"  Lower Barrier: {fifty_percent['lower_barrier']:.4f} ({fifty_percent['lower_barrier']*100:.2f}%)")
    print()

    # 25% barrier prediction
    twenty_five_percent = predictions["twenty_five_percent"]
    print(f"25% Barrier Prediction:")
    print(f"  Confidence: {twenty_five_percent['confidence']:.4f}")
    print(f"  Direction: {twenty_five_percent['direction']}")
    print(f"  Upper Barrier: {twenty_five_percent['upper_barrier']:.4f} ({twenty_five_percent['upper_barrier']*100:.2f}%)")
    print(f"  Lower Barrier: {twenty_five_percent['lower_barrier']:.4f} ({twenty_five_percent['lower_barrier']*100:.2f}%)")
    print()

    # Combined results
    combined_confidence = predictions["combined_confidence"]
    green_light_signal = predictions["green_light_signal"]

    print(f"🎯 Combined Results:")
    print(f"  Combined Confidence: {combined_confidence:.4f}")
    print(f"  Green Light Signal: {green_light_signal['signal']}")
    print(f"  Reason: {green_light_signal['reason']}")
    print()

    # Threshold analysis
    print(f"📊 Threshold Analysis:")
    print(f"  50% Barrier OK: {green_light_signal['fifty_percent_ok']}")
    print(f"  25% Barrier OK: {green_light_signal['twenty_five_percent_ok']}")
    print(f"  Combined OK: {green_light_signal['combined_ok']}")
    print()

    # Test different scenarios
    print("🔄 Testing Different Scenarios:")
    print("-" * 40)

    # Test with higher confidence
    print("Scenario 1: High Confidence")
    high_confidence_predictions = {
        "fifty_percent": {"confidence": 0.85, "direction": "UP"},
        "twenty_five_percent": {"confidence": 0.82, "direction": "UP"}
    }
    high_combined = ml_tactics._calculate_combined_confidence(high_confidence_predictions)
    high_signal = ml_tactics._evaluate_green_light_signal(high_confidence_predictions, high_combined)
    print(f"  Combined Confidence: {high_combined:.4f}")
    print(f"  Signal: {high_signal['signal']}")
    print()

    # Test with low confidence
    print("Scenario 2: Low Confidence")
    low_confidence_predictions = {
        "fifty_percent": {"confidence": 0.45, "direction": "UP"},
        "twenty_five_percent": {"confidence": 0.42, "direction": "UP"}
    }
    low_combined = ml_tactics._calculate_combined_confidence(low_confidence_predictions)
    low_signal = ml_tactics._evaluate_green_light_signal(low_confidence_predictions, low_combined)
    print(f"  Combined Confidence: {low_combined:.4f}")
    print(f"  Signal: {low_signal['signal']}")
    print()

    # Test exit signal evaluation
    print("🚪 Exit Signal Testing:")
    print("-" * 40)

    # Test exit thresholds
    exit_signal = await ml_tactics.evaluate_exit_signal(predictions, {})
    print(f"Current Exit Signal: {exit_signal['exit_signal']}")
    print(f"Reason: {exit_signal['reason']}")
    print()

    print("✅ Multi-output prediction system test completed!")
    return predictions

async def test_integration_with_position_sizing():
    """Test integration with position sizing."""
    print("\n🧪 Testing Integration with Position Sizing")
    print("=" * 60)

    # Generate predictions
    market_data = create_mock_market_data(100)
    analyst_barriers = create_mock_analyst_barriers()
    ml_tactics = MockMLTacticsManager()

    predictions = await ml_tactics.generate_multi_output_predictions(
        market_data=market_data,
        analyst_barriers=analyst_barriers,
        symbol="BTCUSDT",
        timeframe="1m"
    )

    # Simulate position sizing calculation
    combined_confidence = predictions["combined_confidence"]

    # Simple position sizing logic
    if combined_confidence >= 0.8:
        position_size = 0.5  # 50% of account
    elif combined_confidence >= 0.7:
        position_size = 0.3  # 30% of account
    elif combined_confidence >= 0.6:
        position_size = 0.15  # 15% of account
    else:
        position_size = 0.05  # 5% of account

    # Simple leverage calculation
    if combined_confidence >= 0.8:
        leverage = 3.0
    elif combined_confidence >= 0.7:
        leverage = 2.0
    elif combined_confidence >= 0.6:
        leverage = 1.5
    else:
        leverage = 1.0

    print(f"📊 Position Sizing Results:")
    print(f"  Combined Confidence: {combined_confidence:.4f}")
    print(f"  Position Size: {position_size:.2%} of account")
    print(f"  Leverage: {leverage:.1f}x")
    print()

    # Check if we should open position
    green_light = predictions["green_light_signal"]["signal"]
    if green_light == "GREEN_LIGHT":
        print("🟢 GREEN LIGHT: Position can be opened")
        print(f"  Action: {predictions['fifty_percent']['direction']}")
        print(f"  Size: {position_size:.2%} of account")
        print(f"  Leverage: {leverage:.1f}x")
    else:
        print("🔴 RED LIGHT: No position should be opened")

    print("\n✅ Integration test completed!")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_multi_output_system())
    asyncio.run(test_integration_with_position_sizing())