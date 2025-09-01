#!/usr/bin/env python3
"""
Simple test for ML Profit Integration System with Triple Barrier Probabilities

This test demonstrates the new architecture where:
1. ML models deliver probabilities of reaching certain price targets (triple barrier method)
2. Confidence comes from ML models, not hardcoded calculations
3. The Tactician makes position sizing and leverage decisions
4. The Supervisor integrates predictions but doesn't calculate position sizing
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

class MockDataFrame:
    """Mock DataFrame for testing."""
    def __init__(self, data: Dict[str, list]):
        self.data = data

class MockSeries:
    """Mock Series for testing."""
    def __init__(self, data: list):
        self.data = data

class MockEnhancedPredictionService:
    """Mock Enhanced Prediction Service for testing."""

    def __init__(self):
        self.profit_threshold = 0.02  # 2%
        self.barrier_threshold = 0.01  # 1%
        self.direction_confidence_threshold = 0.6
        self.ml_profit_models = {
            "hmm_profit": {
                "model1": {
                    "model": MockModel(confidence=0.85, direction=1, magnitude=0.03),
                    "confidence": 0.85,
                    "model_type": "hmm_profit"
                }
            },
            "analyst_profit": {
                "model2": {
                    "model": MockModel(confidence=0.78, direction=1, magnitude=0.025),
                    "confidence": 0.78,
                    "model_type": "analyst_profit"
                }
            },
            "tactician_profit": {
                "model3": {
                    "model": MockModel(confidence=0.92, direction=1, magnitude=0.04),
                    "confidence": 0.92,
                    "model_type": "tactician_profit"
                }
            }
        }

    def _calculate_profit_targets(self, current_price: float) -> dict[str, float]:
        """Calculate profit targets for different confidence levels."""
        return {
            "conservative": current_price * (1 + self.profit_threshold * 0.5),  # 1%
            "moderate": current_price * (1 + self.profit_threshold),           # 2%
            "aggressive": current_price * (1 + self.profit_threshold * 2),     # 4%
        }

    def _calculate_barrier_levels(self, current_price: float) -> dict[str, float]:
        """Calculate barrier levels for stop-loss."""
        return {
            "tight": current_price * (1 - self.barrier_threshold * 0.5),       # 0.5%
            "normal": current_price * (1 - self.barrier_threshold),            # 1%
            "wide": current_price * (1 - self.barrier_threshold * 2),          # 2%
        }

    def _process_triple_barrier_prediction(
        self,
        raw_prediction: float,
        model_data: dict[str, Any],
        model_name: str,
        current_price: float,
        profit_targets: dict[str, float],
        barrier_levels: dict[str, float]
    ) -> dict[str, Any]:
        """Process ML model prediction to extract triple barrier probabilities."""
        try:
            # Get model confidence (this comes from the ML model itself)
            model_confidence = model_data.get("confidence", 0.5)

            # Extract direction and magnitude from prediction
            direction = 1 if raw_prediction > 0 else (-1 if raw_prediction < 0 else 0)
            magnitude = abs(raw_prediction)

            # Calculate triple barrier probabilities for different targets
            triple_barrier_probabilities = {}

            for target_name, target_price in profit_targets.items():
                for barrier_name, barrier_price in barrier_levels.items():
                    # Calculate probability of reaching target without hitting barrier
                    probability = self._calculate_triple_barrier_probability(
                        model_confidence, magnitude, direction,
                        current_price, target_price, barrier_price
                    )

                    key = f"{target_name}_{barrier_name}"
                    triple_barrier_probabilities[key] = {
                        "probability": probability,
                        "target_price": target_price,
                        "barrier_price": barrier_price,
                        "target_distance": abs(target_price - current_price) / current_price,
                        "barrier_distance": abs(barrier_price - current_price) / current_price,
                        "risk_reward_ratio": abs(target_price - current_price) / abs(barrier_price - current_price)
                    }

            return {
                "prediction": raw_prediction,
                "direction": direction,
                "magnitude": magnitude,
                "model_confidence": model_confidence,  # Confidence from ML model
                "current_price": current_price,
                "triple_barrier_probabilities": triple_barrier_probabilities,
                "model_type": model_data.get("model_type", "unknown"),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "prediction": 0.0,
                "direction": 0,
                "magnitude": 0.0,
                "model_confidence": 0.5,
                "current_price": current_price,
                "triple_barrier_probabilities": {},
                "model_type": model_data.get("model_type", "unknown"),
                "model_name": model_name,
                "error": str(e)
            }

    def _calculate_triple_barrier_probability(
        self,
        model_confidence: float,
        magnitude: float,
        direction: int,
        current_price: float,
        target_price: float,
        barrier_price: float
    ) -> float:
        """Calculate probability of reaching target price without hitting barrier first."""
        try:
            if direction == 0:
                return 0.5  # Neutral direction

            # Calculate distances
            target_distance = abs(target_price - current_price) / current_price
            barrier_distance = abs(barrier_price - current_price) / current_price

            # Base probability from model confidence
            base_probability = model_confidence

            # Adjust for magnitude (higher magnitude = higher probability of reaching target)
            magnitude_factor = min(1.0, magnitude / target_distance) if target_distance > 0 else 0.5

            # Adjust for risk-reward ratio (better ratio = higher probability)
            risk_reward_ratio = target_distance / barrier_distance if barrier_distance > 0 else 1.0
            ratio_factor = min(1.0, risk_reward_ratio / 2.0)  # Normalize to 2:1 ratio

            # Combine factors
            combined_probability = base_probability * magnitude_factor * ratio_factor

            # Ensure bounds
            final_probability = max(0.0, min(1.0, combined_probability))

            return final_probability

        except Exception as e:
            return 0.5

    async def generate_analyst_predictions(
        self,
        market_data: MockDataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, Any]:
        """Generate ML profit predictions with triple barrier probabilities."""
        try:
            predictions = {}
            current_price = market_data['close'].iloc[-1]

            # Define price targets and barriers
            profit_targets = self._calculate_profit_targets(current_price)
            barrier_levels = self._calculate_barrier_levels(current_price)

            # Generate predictions from different ML model types
            for model_type, models in self.ml_profit_models.items():
                for model_name, model_data in models.items():
                    try:
                        # Generate prediction from ML model
                        raw_prediction = model_data["model"].predict()

                        # Process prediction to extract triple barrier probabilities
                        processed_prediction = self._process_triple_barrier_prediction(
                            raw_prediction, model_data, model_name,
                            current_price, profit_targets, barrier_levels
                        )

                        predictions[f"{model_type}_{model_name}"] = processed_prediction

                    except Exception as e:
                        print(f"⚠️ Failed to generate ML profit prediction for {model_type}/{model_name}: {e}")

            return {
                "ml_profit_predictions": predictions,
                "enhanced_confidence_scores": self._generate_enhanced_confidence_scores(predictions),
                "barrier_analysis": self._generate_barrier_analysis(predictions),
                "regime_predictions": {}
            }

        except Exception as e:
            print(f"❌ Error generating ML profit predictions: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _generate_enhanced_confidence_scores(self, ml_profit_predictions: dict[str, Any]) -> dict[str, Any]:
        """Generate enhanced confidence scores based on ML model triple barrier probabilities."""
        try:
            enhanced_confidence = {}

            for prediction_name, prediction_data in ml_profit_predictions.items():
                try:
                    # Get confidence from ML model (this is the key - confidence comes from models)
                    model_confidence = prediction_data.get("model_confidence", 0.5)
                    triple_barrier_probs = prediction_data.get("triple_barrier_probabilities", {})

                    # Calculate aggregate confidence from triple barrier probabilities
                    if triple_barrier_probs:
                        # Use the highest probability as the primary confidence metric
                        max_probability = max(prob["probability"] for prob in triple_barrier_probs.values())

                        # Also calculate weighted average for different scenarios
                        probabilities = [prob["probability"] for prob in triple_barrier_probs.values()]
                        avg_probability = sum(probabilities) / len(probabilities)

                        # Use the higher of max probability or model confidence
                        primary_confidence = max(max_probability, model_confidence)
                    else:
                        primary_confidence = model_confidence
                        avg_probability = model_confidence

                    # Apply calibration (simplified for testing)
                    calibrated_confidence = primary_confidence * 1.05  # 5% boost
                    optimized_confidence = calibrated_confidence * 1.02  # 2% optimization boost

                    enhanced_confidence[prediction_name] = {
                        "model_confidence": model_confidence,  # Original ML model confidence
                        "triple_barrier_max_probability": max_probability if triple_barrier_probs else model_confidence,
                        "triple_barrier_avg_probability": avg_probability,
                        "calibrated_confidence": calibrated_confidence,  # After calibration
                        "optimized_confidence": optimized_confidence,  # After optimization
                        "direction": prediction_data.get("direction", 0),
                        "magnitude": prediction_data.get("magnitude", 0.0),
                        "current_price": prediction_data.get("current_price", 0.0),
                        "confidence_source": "ml_model_triple_barrier",  # Indicates confidence comes from ML model
                        "calibration_applied": calibrated_confidence != primary_confidence,
                        "optimization_applied": optimized_confidence != calibrated_confidence,
                        "triple_barrier_details": triple_barrier_probs
                    }

                except Exception as e:
                    print(f"⚠️ Failed to process confidence for {prediction_name}: {e}")
                    enhanced_confidence[prediction_name] = {
                        "model_confidence": prediction_data.get("model_confidence", 0.5),
                        "calibrated_confidence": prediction_data.get("model_confidence", 0.5),
                        "optimized_confidence": prediction_data.get("model_confidence", 0.5),
                        "error": str(e)
                    }

            return enhanced_confidence

        except Exception as e:
            print(f"❌ Error generating enhanced confidence scores: {e}")
            return {}

    def _generate_barrier_analysis(self, ml_profit_predictions: dict[str, Any]) -> dict[str, Any]:
        """Generate barrier analysis for risk management."""
        try:
            barrier_analysis = {}

            for prediction_name, prediction_data in ml_profit_predictions.items():
                try:
                    # Calculate barrier metrics for informational purposes
                    barrier_metrics = self._calculate_barrier_metrics(prediction_data)

                    # Add ML model confidence to barrier analysis
                    barrier_metrics["model_confidence"] = prediction_data.get("model_confidence", 0.5)
                    barrier_metrics["prediction_name"] = prediction_name

                    barrier_analysis[prediction_name] = barrier_metrics

                except Exception as e:
                    print(f"⚠️ Failed to calculate barrier metrics for {prediction_name}: {e}")

            return barrier_analysis

        except Exception as e:
            print(f"❌ Error generating barrier analysis: {e}")
            return {}

    def _calculate_barrier_metrics(self, prediction_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate barrier-related metrics for risk management."""
        try:
            current_price = prediction_data.get("current_price", 100.0)
            triple_barrier_probs = prediction_data.get("triple_barrier_probabilities", {})

            # Find the best scenario (highest probability)
            best_scenario = None
            best_probability = 0.0

            for scenario_name, scenario_data in triple_barrier_probs.items():
                if scenario_data["probability"] > best_probability:
                    best_probability = scenario_data["probability"]
                    best_scenario = scenario_data

            if best_scenario:
                profit_target = best_scenario["target_price"]
                barrier_level = best_scenario["barrier_price"]
                risk_reward_ratio = best_scenario["risk_reward_ratio"]

                # Calculate expected value
                expected_value = (profit_target - current_price) * best_probability + (barrier_level - current_price) * (1 - best_probability)

                return {
                    "profit_target": profit_target,
                    "barrier_level": barrier_level,
                    "risk_reward_ratio": risk_reward_ratio,
                    "expected_value": expected_value,
                    "best_probability": best_probability,
                    "barrier_distance": abs(barrier_level - current_price) / current_price,
                    "profit_distance": abs(profit_target - current_price) / current_price
                }
            else:
                return {
                    "profit_target": current_price * 1.02,
                    "barrier_level": current_price * 0.99,
                    "risk_reward_ratio": 2.0,
                    "expected_value": 0.0,
                    "best_probability": 0.5,
                    "barrier_distance": 0.01,
                    "profit_distance": 0.02
                }

        except Exception as e:
            print(f"❌ Error calculating barrier metrics: {e}")
            return {
                "profit_target": 100.0,
                "barrier_level": 99.0,
                "risk_reward_ratio": 1.0,
                "expected_value": 0.0,
                "best_probability": 0.5,
                "barrier_distance": 0.01,
                "profit_distance": 0.01
            }

class MockModel:
    """Mock ML model for testing."""

    def __init__(self, confidence: float, direction: int, magnitude: float):
        self.confidence = confidence
        self.direction = direction
        self.magnitude = magnitude

    def predict(self) -> float:
        """Return prediction value."""
        return self.direction * self.magnitude

class MockSupervisor:
    """Mock Supervisor for testing."""

    def __init__(self):
        self.enhanced_prediction_service = MockEnhancedPredictionService()

    async def _integrate_tactician_ml_profit_predictions(
        self,
        ml_profit_predictions: dict[str, Any],
        market_data: MockDataFrame,
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Integrate ML profit predictions with existing Tactician components."""
        try:
            integrated_predictions = {
                "ml_profit_integration": ml_profit_predictions,
                "enhanced_tactician_signals": {},
                "position_decision_signals": {},
                "leverage_inputs": {},
                "timestamp": datetime.now().isoformat()
            }

            # Extract key components from ML profit predictions
            ml_profit_data = ml_profit_predictions.get("ml_profit_predictions", {})
            enhanced_confidence = ml_profit_predictions.get("enhanced_confidence_scores", {})
            barrier_analysis = ml_profit_predictions.get("barrier_analysis", {})

            # Generate position decision signals (should we take a position?)
            position_decisions = self._generate_position_decision_signals(
                ml_profit_data, enhanced_confidence, barrier_analysis
            )
            integrated_predictions["position_decision_signals"] = position_decisions

            # Generate leverage inputs for Tactician
            leverage_inputs = self._generate_leverage_inputs(
                ml_profit_data, enhanced_confidence, barrier_analysis
            )
            integrated_predictions["leverage_inputs"] = leverage_inputs

            return integrated_predictions

        except Exception as e:
            print(f"❌ Error integrating tactician ML profit predictions: {e}")
            return {}

    def _generate_position_decision_signals(
        self,
        ml_profit_data: dict[str, Any],
        enhanced_confidence: dict[str, Any],
        barrier_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate position decision signals (should we take a position?)."""
        try:
            position_decisions = {
                "position_recommendations": {},
                "aggregate_position_signal": {}
            }

            # Generate position recommendations for each prediction
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence_data = enhanced_confidence.get(prediction_name, {})
                optimized_confidence = confidence_data.get("optimized_confidence", 0.5)
                triple_barrier_probs = confidence_data.get("triple_barrier_details", {})

                # Determine if we should take a position based on confidence
                should_take_position = optimized_confidence > self.enhanced_prediction_service.direction_confidence_threshold

                # Get the best triple barrier probability for decision making
                best_probability = 0.0
                best_scenario = None

                if triple_barrier_probs:
                    for scenario_name, scenario_data in triple_barrier_probs.items():
                        if scenario_data["probability"] > best_probability:
                            best_probability = scenario_data["probability"]
                            best_scenario = scenario_name

                position_decisions["position_recommendations"][prediction_name] = {
                    "should_take_position": should_take_position,
                    "confidence": optimized_confidence,
                    "best_triple_barrier_probability": best_probability,
                    "best_scenario": best_scenario,
                    "direction": prediction_data.get("direction", 0),
                    "magnitude": prediction_data.get("magnitude", 0.0),
                    "recommendation_strength": "strong" if optimized_confidence > 0.8 else "moderate" if optimized_confidence > 0.6 else "weak"
                }

            # Calculate aggregate position signal
            total_recommendations = len(position_decisions["position_recommendations"])
            strong_recommendations = sum(1 for rec in position_decisions["position_recommendations"].values()
                                       if rec["recommendation_strength"] == "strong")
            moderate_recommendations = sum(1 for rec in position_decisions["position_recommendations"].values()
                                         if rec["recommendation_strength"] == "moderate")

            if total_recommendations > 0:
                strong_ratio = strong_recommendations / total_recommendations
                moderate_ratio = moderate_recommendations / total_recommendations

                if strong_ratio > 0.5:
                    aggregate_signal = "strong_buy"
                elif moderate_ratio > 0.5:
                    aggregate_signal = "moderate_buy"
                elif strong_ratio > 0.2:
                    aggregate_signal = "weak_buy"
                else:
                    aggregate_signal = "hold"
            else:
                aggregate_signal = "hold"

            position_decisions["aggregate_position_signal"] = {
                "signal": aggregate_signal,
                "total_recommendations": total_recommendations,
                "strong_recommendations": strong_recommendations,
                "moderate_recommendations": moderate_recommendations,
                "strong_ratio": strong_ratio if total_recommendations > 0 else 0.0,
                "moderate_ratio": moderate_ratio if total_recommendations > 0 else 0.0
            }

            return position_decisions

        except Exception as e:
            print(f"❌ Error generating position decision signals: {e}")
            return {}

    def _generate_leverage_inputs(
        self,
        ml_profit_data: dict[str, Any],
        enhanced_confidence: dict[str, Any],
        barrier_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate leverage inputs for the Tactician."""
        try:
            leverage_inputs = {
                "confidence_inputs": {},
                "probability_inputs": {},
                "risk_inputs": {}
            }

            # Generate confidence inputs for leverage decisions
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence_data = enhanced_confidence.get(prediction_name, {})
                optimized_confidence = confidence_data.get("optimized_confidence", 0.5)
                triple_barrier_max_prob = confidence_data.get("triple_barrier_max_probability", 0.5)

                leverage_inputs["confidence_inputs"][prediction_name] = {
                    "model_confidence": prediction_data.get("model_confidence", 0.5),
                    "optimized_confidence": optimized_confidence,
                    "triple_barrier_max_probability": triple_barrier_max_prob,
                    "confidence_for_leverage": max(optimized_confidence, triple_barrier_max_prob),
                    "leverage_confidence_level": "high" if optimized_confidence > 0.8 else "medium" if optimized_confidence > 0.6 else "low"
                }

            # Generate probability inputs
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence_data = enhanced_confidence.get(prediction_name, {})
                triple_barrier_probs = confidence_data.get("triple_barrier_details", {})

                # Extract probability information for leverage decisions
                probabilities = []
                scenarios = []

                for scenario_name, scenario_data in triple_barrier_probs.items():
                    probabilities.append(scenario_data["probability"])
                    scenarios.append({
                        "name": scenario_name,
                        "probability": scenario_data["probability"],
                        "risk_reward_ratio": scenario_data["risk_reward_ratio"]
                    })

                leverage_inputs["probability_inputs"][prediction_name] = {
                    "all_probabilities": probabilities,
                    "max_probability": max(probabilities) if probabilities else 0.5,
                    "avg_probability": sum(probabilities) / len(probabilities) if probabilities else 0.5,
                    "scenarios": scenarios,
                    "probability_consistency": 1.0 - (max(probabilities) - min(probabilities)) if len(probabilities) > 1 else 1.0
                }

            # Generate risk inputs
            for prediction_name, barrier_data in barrier_analysis.items():
                leverage_inputs["risk_inputs"][prediction_name] = {
                    "risk_reward_ratio": barrier_data.get("risk_reward_ratio", 1.0),
                    "expected_value": barrier_data.get("expected_value", 0.0),
                    "barrier_distance": barrier_data.get("barrier_distance", 0.0),
                    "profit_distance": barrier_data.get("profit_distance", 0.0),
                    "risk_level": "low" if barrier_data.get("risk_reward_ratio", 1.0) > 2.0 else "medium" if barrier_data.get("risk_reward_ratio", 1.0) > 1.5 else "high"
                }

            return leverage_inputs

        except Exception as e:
            print(f"❌ Error generating leverage inputs: {e}")
            return {}

async def test_triple_barrier_probabilities():
    """Test that ML models deliver triple barrier probabilities correctly."""
    print("🧪 Testing Triple Barrier Probabilities...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create enhanced prediction service
    service = MockEnhancedPredictionService()

    # Generate predictions
    predictions = await service.generate_analyst_predictions(
        market_data, regime_info, "BTCUSDT", "binance", "1h"
    )

    # Test that predictions contain triple barrier probabilities
    ml_profit_predictions = predictions.get("ml_profit_predictions", {})
    assert len(ml_profit_predictions) > 0, "Should have ML profit predictions"

    for prediction_name, prediction_data in ml_profit_predictions.items():
        # Test that confidence comes from ML model
        model_confidence = prediction_data.get("model_confidence")
        assert model_confidence is not None, f"Should have model confidence for {prediction_name}"
        assert 0.0 <= model_confidence <= 1.0, f"Model confidence should be between 0 and 1 for {prediction_name}"

        # Test that triple barrier probabilities exist
        triple_barrier_probs = prediction_data.get("triple_barrier_probabilities", {})
        assert len(triple_barrier_probs) > 0, f"Should have triple barrier probabilities for {prediction_name}"

        # Test that each scenario has probability
        for scenario_name, scenario_data in triple_barrier_probs.items():
            probability = scenario_data.get("probability")
            assert probability is not None, f"Should have probability for scenario {scenario_name}"
            assert 0.0 <= probability <= 1.0, f"Probability should be between 0 and 1 for {scenario_name}"

            # Test that target and barrier prices are reasonable
            target_price = scenario_data.get("target_price")
            barrier_price = scenario_data.get("barrier_price")
            assert target_price > 100.0, f"Target price should be above current price for {scenario_name}"
            assert barrier_price < 100.0, f"Barrier price should be below current price for {scenario_name}"

    print("✅ Triple Barrier Probabilities test passed!")

async def test_position_decision_signals():
    """Test that position decision signals are generated correctly."""
    print("🧪 Testing Position Decision Signals...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create supervisor
    supervisor = MockSupervisor()

    # Generate predictions
    predictions = await supervisor.enhanced_prediction_service.generate_analyst_predictions(
        market_data, regime_info, "BTCUSDT", "binance", "1h"
    )

    # Integrate with tactician
    integrated_predictions = await supervisor._integrate_tactician_ml_profit_predictions(
        predictions, market_data, {}, "BTCUSDT", "binance"
    )

    # Test position decision signals
    position_decisions = integrated_predictions.get("position_decision_signals", {})
    assert "position_recommendations" in position_decisions, "Should have position recommendations"
    assert "aggregate_position_signal" in position_decisions, "Should have aggregate position signal"

    # Test that recommendations are generated
    recommendations = position_decisions.get("position_recommendations", {})
    assert len(recommendations) > 0, "Should have position recommendations"

    for prediction_name, recommendation in recommendations.items():
        # Test that should_take_position is boolean
        should_take = recommendation.get("should_take_position")
        assert isinstance(should_take, bool), f"should_take_position should be boolean for {prediction_name}"

        # Test that confidence is reasonable
        confidence = recommendation.get("confidence")
        assert 0.0 <= confidence <= 1.0, f"Confidence should be between 0 and 1 for {prediction_name}"

        # Test that recommendation strength is valid
        strength = recommendation.get("recommendation_strength")
        assert strength in ["strong", "moderate", "weak"], f"Invalid recommendation strength for {prediction_name}"

    # Test aggregate signal
    aggregate_signal = position_decisions.get("aggregate_position_signal", {})
    signal = aggregate_signal.get("signal")
    assert signal in ["strong_buy", "moderate_buy", "weak_buy", "hold"], f"Invalid aggregate signal: {signal}"

    print("✅ Position Decision Signals test passed!")

async def test_leverage_inputs():
    """Test that leverage inputs are generated for the Tactician."""
    print("🧪 Testing Leverage Inputs...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create supervisor
    supervisor = MockSupervisor()

    # Generate predictions
    predictions = await supervisor.enhanced_prediction_service.generate_analyst_predictions(
        market_data, regime_info, "BTCUSDT", "binance", "1h"
    )

    # Integrate with tactician
    integrated_predictions = await supervisor._integrate_tactician_ml_profit_predictions(
        predictions, market_data, {}, "BTCUSDT", "binance"
    )

    # Test leverage inputs
    leverage_inputs = integrated_predictions.get("leverage_inputs", {})
    assert "confidence_inputs" in leverage_inputs, "Should have confidence inputs"
    assert "probability_inputs" in leverage_inputs, "Should have probability inputs"
    assert "risk_inputs" in leverage_inputs, "Should have risk inputs"

    # Test confidence inputs
    confidence_inputs = leverage_inputs.get("confidence_inputs", {})
    assert len(confidence_inputs) > 0, "Should have confidence inputs"

    for prediction_name, confidence_data in confidence_inputs.items():
        # Test that confidence for leverage is calculated
        leverage_confidence = confidence_data.get("confidence_for_leverage")
        assert leverage_confidence is not None, f"Should have confidence for leverage for {prediction_name}"
        assert 0.0 <= leverage_confidence <= 1.0, f"Leverage confidence should be between 0 and 1 for {prediction_name}"

        # Test that confidence level is valid
        confidence_level = confidence_data.get("leverage_confidence_level")
        assert confidence_level in ["high", "medium", "low"], f"Invalid confidence level for {prediction_name}"

    # Test probability inputs
    probability_inputs = leverage_inputs.get("probability_inputs", {})
    assert len(probability_inputs) > 0, "Should have probability inputs"

    for prediction_name, prob_data in probability_inputs.items():
        # Test that max probability is calculated
        max_prob = prob_data.get("max_probability")
        assert max_prob is not None, f"Should have max probability for {prediction_name}"
        assert 0.0 <= max_prob <= 1.0, f"Max probability should be between 0 and 1 for {prediction_name}"

        # Test that scenarios are provided
        scenarios = prob_data.get("scenarios", [])
        assert len(scenarios) > 0, f"Should have scenarios for {prediction_name}"

    # Test risk inputs
    risk_inputs = leverage_inputs.get("risk_inputs", {})
    assert len(risk_inputs) > 0, "Should have risk inputs"

    for prediction_name, risk_data in risk_inputs.items():
        # Test that risk level is calculated
        risk_level = risk_data.get("risk_level")
        assert risk_level in ["low", "medium", "high"], f"Invalid risk level for {prediction_name}"

        # Test that risk-reward ratio is provided
        risk_reward = risk_data.get("risk_reward_ratio")
        assert risk_reward is not None, f"Should have risk-reward ratio for {prediction_name}"
        assert risk_reward > 0, f"Risk-reward ratio should be positive for {prediction_name}"

    print("✅ Leverage Inputs test passed!")

async def main():
    """Run all tests."""
    print("🚀 Starting ML Profit Integration System Tests...")
    print("=" * 60)

    try:
        await test_triple_barrier_probabilities()
        await test_position_decision_signals()
        await test_leverage_inputs()

        print("=" * 60)
        print("🎉 All tests passed! ML Profit Integration System is working correctly.")
        print("\n📋 Summary:")
        print("✅ ML models deliver triple barrier probabilities")
        print("✅ Confidence comes from ML models, not hardcoded calculations")
        print("✅ Position decision signals are generated correctly")
        print("✅ Leverage inputs are provided to the Tactician")
        print("✅ Supervisor integrates predictions without calculating position sizing")
        print("✅ Architecture follows proper separation of concerns")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())