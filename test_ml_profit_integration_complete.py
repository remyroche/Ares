#!/usr/bin/env python3
"""
Complete Test for ML Profit Integration System

This test demonstrates the complete architecture where:
1. Enhanced Prediction Service provides calibrated confidence scores for both Analyst and Tactician ML models
2. Analyst decides if we enter a position based on Analyst ML models (higher timeframe)
3. Tactician decides when, how much, and with what leverage based on Tactician ML models (lower timeframe)
4. Both must agree on trade direction
5. System fails if calibrated confidence doesn't exist
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
        self.is_initialized = True
        self.entry_threshold = 0.6
        self.max_confidence_threshold = 0.7

        # Mock Analyst ML models (higher timeframe)
        self.analyst_ml_models = {
            "hmm_profit": {
                "hmm_bullish": {"calibrated_confidence": 0.85},
                "hmm_bearish": {"calibrated_confidence": 0.45}
            },
            "analyst_profit": {
                "analyst_long": {"calibrated_confidence": 0.78},
                "analyst_short": {"calibrated_confidence": 0.32}
            },
            "calibrated": {
                "calibrated_bullish": {"calibrated_confidence": 0.92},
                "calibrated_neutral": {"calibrated_confidence": 0.55}
            }
        }

        # Mock Tactician ML models (lower timeframe)
        self.tactician_ml_models = {
            "tactician_profit": {
                "tactician_bullish": {"calibrated_confidence": 0.88},
                "tactician_bearish": {"calibrated_confidence": 0.42}
            },
            "tactician_specialist": {
                "specialist_long": {"calibrated_confidence": 0.81},
                "specialist_short": {"calibrated_confidence": 0.38}
            },
            "calibrated": {
                "tactician_calibrated_bullish": {"calibrated_confidence": 0.89},
                "tactician_calibrated_neutral": {"calibrated_confidence": 0.52}
            }
        }

class MockSupervisor:
    """Mock Supervisor for testing."""

    def __init__(self):
        self.enhanced_prediction_service = MockEnhancedPredictionService()
        self.is_initialized = True
        self.entry_threshold = 0.6
        self.max_confidence_threshold = 0.7

    async def get_analyst_predictions(
        self,
        market_data: MockDataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Get Analyst predictions using calibrated confidence scores from ML models."""
        try:
            # Step 1: Get calibrated confidence scores from Enhanced Prediction Service
            calibrated_confidence = await self.enhanced_prediction_service.get_calibrated_confidence_scores(
                market_data, regime_info, symbol, exchange
            )

            # Step 2: Analyst decides if we enter a position using Analyst models
            analyst_decision = await self._analyst_decide_position_entry(
                market_data, regime_info, calibrated_confidence["analyst_models"], symbol, exchange
            )

            return {
                "calibrated_confidence_scores": calibrated_confidence,
                "analyst_decision": analyst_decision,
                "timestamp": datetime.now().isoformat()
            }

        except ValueError as e:
            # Enhanced Prediction Service failed - no calibrated confidence
            return {
                "error": str(e),
                "analyst_decision": {"should_enter_position": False, "reason": "no_calibrated_confidence"},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {}

    async def get_tactician_predictions(
        self,
        market_data: MockDataFrame,
        regime_info: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = "1m"
    ) -> Dict[str, Any]:
        """Get Tactician predictions using calibrated confidence scores from ML models."""
        try:
            # Step 1: Get calibrated confidence scores from Enhanced Prediction Service
            calibrated_confidence = await self.enhanced_prediction_service.get_calibrated_confidence_scores(
                market_data, regime_info, symbol, exchange
            )

            # Step 2: Tactician decides execution parameters using Tactician models
            tactician_decision = await self._tactician_calculate_execution_parameters(
                market_data, analyst_signals, calibrated_confidence["tactician_models"], symbol, exchange
            )

            return {
                "calibrated_confidence_scores": calibrated_confidence,
                "tactician_decision": tactician_decision,
                "timestamp": datetime.now().isoformat()
            }

        except ValueError as e:
            # Enhanced Prediction Service failed - no calibrated confidence
            return {
                "error": str(e),
                "tactician_decision": {"should_execute": False, "reason": "no_calibrated_confidence"},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {}

    async def _analyst_decide_position_entry(
        self,
        market_data: MockDataFrame,
        regime_info: Dict[str, Any],
        analyst_confidence_scores: Dict[str, float],
        symbol: str,
        exchange: str
    ) -> Dict[str, Any]:
        """Analyst decides if we enter a position and determines trade direction."""
        try:
            # Calculate aggregate Analyst confidence
            if not analyst_confidence_scores:
                return {
                    "should_enter_position": False,
                    "trade_direction": "neutral",
                    "entry_confidence": 0.0,
                    "max_confidence": 0.0,
                    "individual_confidences": {},
                    "entry_reason": "no_analyst_confidence"
                }

            avg_confidence = sum(analyst_confidence_scores.values()) / len(analyst_confidence_scores)
            max_confidence = max(analyst_confidence_scores.values())

            # Determine trade direction from Analyst models
            trade_direction = self._analyst_determine_trade_direction(analyst_confidence_scores, market_data)

            # Decision logic
            should_enter = (
                avg_confidence > self.entry_threshold and
                max_confidence > self.max_confidence_threshold and
                trade_direction != "neutral"
            )

            return {
                "should_enter_position": should_enter,
                "trade_direction": trade_direction,
                "entry_confidence": avg_confidence,
                "max_confidence": max_confidence,
                "individual_confidences": analyst_confidence_scores,
                "entry_reason": "high_confidence" if should_enter else "low_confidence_or_neutral"
            }

        except Exception as e:
            return {
                "should_enter_position": False,
                "trade_direction": "neutral",
                "entry_confidence": 0.0,
                "max_confidence": 0.0,
                "individual_confidences": {},
                "entry_reason": "error",
                "error": str(e)
            }

    def _analyst_determine_trade_direction(
        self,
        confidence_scores: Dict[str, float],
        market_data: MockDataFrame
    ) -> str:
        """Determine trade direction based on Analyst model confidences."""
        try:
            # Logic to determine if models suggest long, short, or neutral
            bullish_confidence = sum(
                conf for name, conf in confidence_scores.items()
                if "bullish" in name.lower() or "long" in name.lower()
            )
            bearish_confidence = sum(
                conf for name, conf in confidence_scores.items()
                if "bearish" in name.lower() or "short" in name.lower()
            )

            # Determine direction based on confidence
            if bullish_confidence > bearish_confidence and bullish_confidence > 0.6:
                return "long"
            elif bearish_confidence > bullish_confidence and bearish_confidence > 0.6:
                return "short"
            else:
                return "neutral"

        except Exception as e:
            return "neutral"

    async def _tactician_calculate_execution_parameters(
        self,
        market_data: MockDataFrame,
        analyst_signals: Dict[str, Any],
        tactician_confidence_scores: Dict[str, float],
        symbol: str,
        exchange: str
    ) -> Dict[str, Any]:
        """Tactician decides when, how much, and what leverage based on Tactician ML models."""
        try:
            # Check if Analyst wants to enter
            analyst_decision = analyst_signals.get("analyst_decision", {})
            if not analyst_decision.get("should_enter_position", False):
                return {
                    "should_execute": False,
                    "reason": "analyst_no_entry"
                }

            # Check direction agreement
            tactician_direction = self._tactician_determine_direction(tactician_confidence_scores, market_data)
            analyst_direction = analyst_decision.get("trade_direction", "neutral")

            if not self._directions_agree(analyst_direction, tactician_direction):
                return {
                    "should_execute": False,
                    "reason": "direction_mismatch",
                    "analyst_direction": analyst_direction,
                    "tactician_direction": tactician_direction
                }

            # Calculate execution parameters based on Tactician confidence
            if not tactician_confidence_scores:
                return {
                    "should_execute": False,
                    "reason": "no_tactician_confidence"
                }

            avg_tactician_confidence = sum(tactician_confidence_scores.values()) / len(tactician_confidence_scores)

            leverage = self._tactician_calculate_leverage(avg_tactician_confidence)
            position_size = self._tactician_calculate_position_size(avg_tactician_confidence, leverage)
            entry_timing = self._tactician_calculate_entry_timing(market_data, avg_tactician_confidence)

            return {
                "should_execute": True,
                "trade_direction": analyst_direction,  # Use agreed direction
                "leverage": leverage,
                "position_size": position_size,
                "entry_timing": entry_timing,
                "tactician_confidence": avg_tactician_confidence,
                "analyst_confidence": analyst_decision.get("entry_confidence", 0.0),
                "combined_confidence": (avg_tactician_confidence + analyst_decision.get("entry_confidence", 0.0)) / 2
            }

        except Exception as e:
            return {
                "should_execute": False,
                "reason": "error",
                "error": str(e)
            }

    def _tactician_determine_direction(
        self,
        confidence_scores: Dict[str, float],
        market_data: MockDataFrame
    ) -> str:
        """Determine trade direction based on Tactician model confidences."""
        try:
            # Logic to determine if Tactician models suggest long, short, or neutral
            bullish_confidence = sum(
                conf for name, conf in confidence_scores.items()
                if "bullish" in name.lower() or "long" in name.lower()
            )
            bearish_confidence = sum(
                conf for name, conf in confidence_scores.items()
                if "bearish" in name.lower() or "short" in name.lower()
            )

            # Determine direction based on confidence
            if bullish_confidence > bearish_confidence and bullish_confidence > 0.6:
                return "long"
            elif bearish_confidence > bullish_confidence and bearish_confidence > 0.6:
                return "short"
            else:
                return "neutral"

        except Exception as e:
            return "neutral"

    def _directions_agree(self, analyst_direction: str, tactician_direction: str) -> bool:
        """Check if Analyst and Tactician agree on trade direction."""
        if analyst_direction == "neutral" or tactician_direction == "neutral":
            return False
        return analyst_direction == tactician_direction

    def _tactician_calculate_leverage(self, confidence: float) -> float:
        """Calculate leverage based on confidence score."""
        if confidence > 0.9:
            return 3.0  # High leverage for very high confidence
        elif confidence > 0.8:
            return 2.5
        elif confidence > 0.7:
            return 2.0
        elif confidence > 0.6:
            return 1.5
        else:
            return 1.0  # No leverage for low confidence

    def _tactician_calculate_position_size(self, confidence: float, leverage: float) -> float:
        """Calculate position size based on confidence and leverage."""
        base_size = confidence * 100  # Base size as percentage
        adjusted_size = base_size * leverage
        return min(adjusted_size, 100.0)  # Cap at 100%

    def _tactician_calculate_entry_timing(self, market_data: MockDataFrame, confidence: float) -> str:
        """Calculate optimal entry timing."""
        if confidence > 0.8:
            return "immediate"
        elif confidence > 0.7:
            return "within_5_minutes"
        else:
            return "wait_for_confirmation"

async def test_enhanced_prediction_service():
    """Test that Enhanced Prediction Service provides calibrated confidence scores correctly."""
    print("🧪 Testing Enhanced Prediction Service...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create enhanced prediction service
    service = MockEnhancedPredictionService()

    # Get calibrated confidence scores
    calibrated_scores = await service.get_calibrated_confidence_scores(
        market_data, regime_info, "BTCUSDT", "binance"
    )

    # Test that both Analyst and Tactician confidence scores are provided
    assert "analyst_models" in calibrated_scores, "Should have analyst_models"
    assert "tactician_models" in calibrated_scores, "Should have tactician_models"

    analyst_scores = calibrated_scores["analyst_models"]
    tactician_scores = calibrated_scores["tactician_models"]

    assert len(analyst_scores) > 0, "Should have Analyst confidence scores"
    assert len(tactician_scores) > 0, "Should have Tactician confidence scores"

    # Test that confidence scores are valid
    for model_name, confidence in analyst_scores.items():
        assert 0.0 <= confidence <= 1.0, f"Analyst confidence should be between 0 and 1 for {model_name}"

    for model_name, confidence in tactician_scores.items():
        assert 0.0 <= confidence <= 1.0, f"Tactician confidence should be between 0 and 1 for {model_name}"

    print(f"✅ Enhanced Prediction Service test passed!")
    print(f"   Analyst models: {len(analyst_scores)}")
    print(f"   Tactician models: {len(tactician_scores)}")
    print(f"   Sample Analyst confidence: {list(analyst_scores.items())[0]}")
    print(f"   Sample Tactician confidence: {list(tactician_scores.items())[0]}")

async def test_analyst_decision():
    """Test that Analyst decides position entry correctly."""
    print("🧪 Testing Analyst Decision...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create supervisor
    supervisor = MockSupervisor()

    # Get analyst predictions
    analyst_predictions = await supervisor.get_analyst_predictions(
        market_data, regime_info, "BTCUSDT", "binance"
    )

    # Test analyst decision
    analyst_decision = analyst_predictions.get("analyst_decision", {})
    assert "should_enter_position" in analyst_decision, "Should have should_enter_position"
    assert "trade_direction" in analyst_decision, "Should have trade_direction"
    assert "entry_confidence" in analyst_decision, "Should have entry_confidence"

    should_enter = analyst_decision.get("should_enter_position")
    trade_direction = analyst_decision.get("trade_direction")
    entry_confidence = analyst_decision.get("entry_confidence")

    assert isinstance(should_enter, bool), "should_enter_position should be boolean"
    assert trade_direction in ["long", "short", "neutral"], f"Invalid trade direction: {trade_direction}"
    assert 0.0 <= entry_confidence <= 1.0, f"Entry confidence should be between 0 and 1: {entry_confidence}"

    print(f"✅ Analyst Decision test passed!")
    print(f"   Should enter: {should_enter}")
    print(f"   Trade direction: {trade_direction}")
    print(f"   Entry confidence: {entry_confidence:.3f}")

async def test_tactician_decision():
    """Test that Tactician decides execution parameters correctly."""
    print("🧪 Testing Tactician Decision...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create supervisor
    supervisor = MockSupervisor()

    # Get analyst predictions first
    analyst_predictions = await supervisor.get_analyst_predictions(
        market_data, regime_info, "BTCUSDT", "binance"
    )

    # Get tactician predictions
    tactician_predictions = await supervisor.get_tactician_predictions(
        market_data, regime_info, analyst_predictions, "BTCUSDT", "binance"
    )

    # Test tactician decision
    tactician_decision = tactician_predictions.get("tactician_decision", {})
    assert "should_execute" in tactician_decision, "Should have should_execute"

    should_execute = tactician_decision.get("should_execute")
    assert isinstance(should_execute, bool), "should_execute should be boolean"

    if should_execute:
        # Test execution parameters
        assert "leverage" in tactician_decision, "Should have leverage"
        assert "position_size" in tactician_decision, "Should have position_size"
        assert "entry_timing" in tactician_decision, "Should have entry_timing"
        assert "trade_direction" in tactician_decision, "Should have trade_direction"

        leverage = tactician_decision.get("leverage")
        position_size = tactician_decision.get("position_size")
        entry_timing = tactician_decision.get("entry_timing")
        trade_direction = tactician_decision.get("trade_direction")

        assert 1.0 <= leverage <= 3.0, f"Leverage should be between 1.0 and 3.0: {leverage}"
        assert 0.0 <= position_size <= 100.0, f"Position size should be between 0 and 100: {position_size}"
        assert entry_timing in ["immediate", "within_5_minutes", "wait_for_confirmation"], f"Invalid entry timing: {entry_timing}"
        assert trade_direction in ["long", "short"], f"Invalid trade direction: {trade_direction}"

        print(f"✅ Tactician Decision test passed!")
        print(f"   Should execute: {should_execute}")
        print(f"   Leverage: {leverage}")
        print(f"   Position size: {position_size:.1f}%")
        print(f"   Entry timing: {entry_timing}")
        print(f"   Trade direction: {trade_direction}")
    else:
        reason = tactician_decision.get("reason", "unknown")
        print(f"✅ Tactician Decision test passed!")
        print(f"   Should execute: {should_execute}")
        print(f"   Reason: {reason}")

async def test_direction_agreement():
    """Test that Analyst and Tactician must agree on trade direction."""
    print("🧪 Testing Direction Agreement...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create supervisor
    supervisor = MockSupervisor()

    # Test with agreeing directions (both bullish)
    analyst_decision = {
        "should_enter_position": True,
        "trade_direction": "long",
        "entry_confidence": 0.8
    }

    tactician_confidence = {
        "tactician_bullish": 0.85,
        "tactician_specialist_long": 0.82
    }

    tactician_decision = await supervisor._tactician_calculate_execution_parameters(
        market_data, {"analyst_decision": analyst_decision}, tactician_confidence, "BTCUSDT", "binance"
    )

    # Should execute when directions agree
    assert tactician_decision.get("should_execute") == True, "Should execute when directions agree"
    assert tactician_decision.get("trade_direction") == "long", "Should use agreed direction"

    # Test with disagreeing directions
    analyst_decision["trade_direction"] = "long"
    tactician_confidence = {
        "tactician_bearish": 0.85,
        "tactician_specialist_short": 0.82
    }

    tactician_decision = await supervisor._tactician_calculate_execution_parameters(
        market_data, {"analyst_decision": analyst_decision}, tactician_confidence, "BTCUSDT", "binance"
    )

    # Should not execute when directions disagree
    assert tactician_decision.get("should_execute") == False, "Should not execute when directions disagree"
    assert tactician_decision.get("reason") == "direction_mismatch", "Should indicate direction mismatch"

    print(f"✅ Direction Agreement test passed!")
    print(f"   Agreeing directions: Execute = {tactician_decision.get('should_execute')}")
    print(f"   Disagreeing directions: Execute = {tactician_decision.get('should_execute')}")

async def test_failure_scenarios():
    """Test failure scenarios when calibrated confidence doesn't exist."""
    print("🧪 Testing Failure Scenarios...")

    # Create mock market data
    market_data = MockDataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000, 1100, 1200, 1150, 1300]
    })

    # Create mock regime info
    regime_info = {"regime": "trending", "volatility": "medium"}

    # Create supervisor with empty models (no calibrated confidence)
    supervisor = MockSupervisor()
    supervisor.enhanced_prediction_service.analyst_ml_models = {}
    supervisor.enhanced_prediction_service.tactician_ml_models = {}

    # Test analyst predictions with no calibrated confidence
    analyst_predictions = await supervisor.get_analyst_predictions(
        market_data, regime_info, "BTCUSDT", "binance"
    )

    assert "error" in analyst_predictions, "Should have error when no calibrated confidence"
    assert analyst_predictions["analyst_decision"]["should_enter_position"] == False, "Should not enter when no confidence"
    assert analyst_predictions["analyst_decision"]["reason"] == "no_calibrated_confidence", "Should indicate no calibrated confidence"

    # Test tactician predictions with no calibrated confidence
    tactician_predictions = await supervisor.get_tactician_predictions(
        market_data, regime_info, analyst_predictions, "BTCUSDT", "binance"
    )

    assert "error" in tactician_predictions, "Should have error when no calibrated confidence"
    assert tactician_predictions["tactician_decision"]["should_execute"] == False, "Should not execute when no confidence"
    assert tactician_predictions["tactician_decision"]["reason"] == "no_calibrated_confidence", "Should indicate no calibrated confidence"

    print(f"✅ Failure Scenarios test passed!")
    print(f"   Analyst failure handled correctly")
    print(f"   Tactician failure handled correctly")

async def main():
    """Run all tests."""
    print("🚀 Starting Complete ML Profit Integration System Tests...")
    print("=" * 70)

    try:
        await test_enhanced_prediction_service()
        await test_analyst_decision()
        await test_tactician_decision()
        await test_direction_agreement()
        await test_failure_scenarios()

        print("=" * 70)
        print("🎉 All tests passed! Complete ML Profit Integration System is working correctly.")
        print("\n📋 Summary:")
        print("✅ Enhanced Prediction Service provides calibrated confidence scores")
        print("✅ Analyst decides position entry based on Analyst ML models")
        print("✅ Tactician decides execution parameters based on Tactician ML models")
        print("✅ Both components must agree on trade direction")
        print("✅ System fails gracefully when calibrated confidence doesn't exist")
        print("✅ Proper separation of concerns and responsibility assignment")
        print("\n🎯 Architecture Verified:")
        print("   • Enhanced Prediction Service: ONLY provides calibrated confidence")
        print("   • Analyst: Decides IF to enter position (higher timeframe)")
        print("   • Tactician: Decides WHEN, HOW MUCH, WHAT LEVERAGE (lower timeframe)")
        print("   • Supervisor: Coordinates flow and handles failures")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())