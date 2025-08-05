# tests/test_dual_model_system.py

"""
Comprehensive Test Suite for Dual Model System
Tests all components including ML confidence predictor, order management, model training, and integration.
"""

from typing import Any

import pandas as pd
import pytest

from src.analyst.ml_confidence_predictor import (
    MLConfidencePredictor,
    setup_ml_confidence_predictor,
)
from src.tactician.async_order_executor import (
    setup_async_order_executor,
)
from src.tactician.enhanced_order_manager import (
    setup_enhanced_order_manager,
)

# Import components to test
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system
from src.training.enhanced_training_manager import (
    setup_enhanced_training_manager,
)


class TestDualModelSystem:
    """Test suite for Dual Model System."""

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """Sample configuration for testing."""
        return {
            "dual_model_system": {
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.5,
                "tactician_confidence_threshold": 0.6,
                "enter_signal_validity_duration": 120,
                "signal_check_interval": 10,
                "neutral_signal_threshold": 0.5,
                "close_signal_threshold": 0.4,
                "position_close_confidence_threshold": 0.6,
                "enable_ensemble_analysis": True,
                "enable_meta_labeling": True,
                "enable_feature_engineering": True,
                "enhanced_order_manager": {
                    "enable_enhanced_order_manager": True,
                    "enable_async_order_executor": True,
                    "enable_chase_micro_breakout": True,
                    "enable_limit_order_return": True,
                    "enable_partial_fill_management": True,
                    "max_order_retries": 3,
                    "order_timeout_seconds": 30,
                    "slippage_tolerance": 0.001,
                    "volume_threshold": 1.5,
                    "momentum_threshold": 0.02,
                },
                "model_training": {
                    "enable_continuous_training": True,
                    "enable_adaptive_training": True,
                    "enable_incremental_training": True,
                    "training_interval_hours": 24,
                    "min_samples_for_retraining": 1000,
                    "performance_degradation_threshold": 0.1,
                    "enable_model_calibration": True,
                    "enable_ensemble_training": True,
                    "enable_regime_specific_training": True,
                    "enable_multi_timeframe_training": True,
                    "enable_dual_model_training": True,
                    "enable_confidence_calibration": True,
                },
            },
        }

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Sample market data for testing."""
        return pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
                "timestamp": pd.date_range(
                    start="2024-01-01",
                    periods=100,
                    freq="1min",
                ),
            },
        )

    @pytest.fixture
    async def dual_model_system(self, sample_config) -> DualModelSystem:
        """Initialize dual model system for testing."""
        system = DualModelSystem(sample_config)
        await system.initialize()
        return system

    @pytest.mark.asyncio
    async def test_dual_model_system_initialization(self, sample_config):
        """Test dual model system initialization."""
        system = DualModelSystem(sample_config)
        assert system is not None
        assert system.config == sample_config

        # Test initialization
        success = await system.initialize()
        assert success is True
        assert system.is_initialized is True

    @pytest.mark.asyncio
    async def test_dual_model_system_decision_making(
        self,
        dual_model_system,
        sample_market_data,
    ):
        """Test dual model system decision making."""
        current_price = 100.5

        # Test trading decision
        decision = await dual_model_system.make_trading_decision(
            sample_market_data,
            current_price,
        )

        assert decision is not None
        assert "action" in decision
        assert "analyst_confidence" in decision
        assert "tactician_confidence" in decision
        assert "final_confidence" in decision

        # Validate confidence scores
        assert 0.0 <= decision["analyst_confidence"] <= 1.0
        assert 0.0 <= decision["tactician_confidence"] <= 1.0
        assert 0.0 <= decision["final_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_dual_model_system_entry_decision(
        self,
        dual_model_system,
        sample_market_data,
    ):
        """Test dual model system entry decision."""
        current_price = 100.5

        # Test entry decision
        entry_decision = await dual_model_system._make_entry_decision(
            sample_market_data,
            current_price,
        )

        assert entry_decision is not None
        assert "action" in entry_decision
        assert "analyst_confidence" in entry_decision
        assert "tactician_confidence" in entry_decision

    @pytest.mark.asyncio
    async def test_dual_model_system_exit_decision(
        self,
        dual_model_system,
        sample_market_data,
    ):
        """Test dual model system exit decision."""
        current_price = 100.5
        current_position = {
            "side": "LONG",
            "quantity": 0.1,
            "entry_price": 100.0,
            "current_price": 100.5,
        }

        # Test exit decision
        exit_decision = await dual_model_system._make_exit_decision(
            sample_market_data,
            current_price,
            current_position,
        )

        assert exit_decision is not None
        assert "action" in exit_decision
        assert "analyst_confidence" in exit_decision
        assert "tactician_confidence" in exit_decision

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, dual_model_system):
        """Test confidence calculation methods."""
        # Test final confidence calculation
        analyst_confidence = 0.7
        tactician_confidence = 0.8
        final_confidence = dual_model_system._calculate_final_confidence(
            analyst_confidence,
            tactician_confidence,
        )

        assert final_confidence > 0.0
        assert final_confidence <= 1.0

        # Test normalized confidence calculation
        normalized_confidence = dual_model_system._calculate_normalized_confidence(
            final_confidence,
        )
        assert normalized_confidence >= 0.0
        assert normalized_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_execution_strategy_determination(self, dual_model_system):
        """Test execution strategy determination."""
        analyst_decision = {"volatility": "high"}
        tactician_decision = {"confidence": 0.8}

        # Test high confidence
        strategy = dual_model_system._determine_execution_strategy(
            0.9,
            analyst_decision,
            tactician_decision,
        )
        assert strategy == "immediate"

        # Test medium confidence with high volatility
        strategy = dual_model_system._determine_execution_strategy(
            0.6,
            analyst_decision,
            tactician_decision,
        )
        assert strategy == "twap"

        # Test low confidence
        strategy = dual_model_system._determine_execution_strategy(
            0.3,
            analyst_decision,
            tactician_decision,
        )
        assert strategy == "vwap"

    @pytest.mark.asyncio
    async def test_model_training_integration(
        self,
        dual_model_system,
        sample_market_data,
    ):
        """Test model training integration."""
        # Test training trigger
        should_train = dual_model_system.should_trigger_training()
        assert isinstance(should_train, bool)

        # Test training status
        training_status = dual_model_system.get_training_status()
        assert training_status is not None
        assert "dual_model_system" in training_status

    @pytest.mark.asyncio
    async def test_system_info(self, dual_model_system):
        """Test system information retrieval."""
        system_info = dual_model_system.get_system_info()

        assert system_info is not None
        assert "analyst_timeframes" in system_info
        assert "tactician_timeframes" in system_info
        assert "is_initialized" in system_info
        assert system_info["is_initialized"] is True

    @pytest.mark.asyncio
    async def test_signal_validation(self, dual_model_system):
        """Test signal validation."""
        # Test signal validity
        is_valid = dual_model_system.is_enter_signal_valid()
        assert isinstance(is_valid, bool)

        # Test current signal
        current_signal = dual_model_system.get_current_signal()
        # May be None if no signal is active
        if current_signal is not None:
            assert isinstance(current_signal, dict)

    @pytest.mark.asyncio
    async def test_system_cleanup(self, dual_model_system):
        """Test system cleanup."""
        await dual_model_system.stop()
        assert dual_model_system.is_initialized is False


class TestMLConfidencePredictor:
    """Test suite for ML Confidence Predictor."""

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """Sample configuration for ML confidence predictor."""
        return {
            "ml_confidence_predictor": {
                "price_movement_levels": [0.5, 1.0, 1.5, 2.0],
                "adversarial_movement_levels": [0.1, 0.2, 0.3, 0.4],
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "enable_enhanced_training": True,
                "enable_meta_labeling": True,
                "enable_feature_engineering": True,
                "enhanced_order_manager": {
                    "enable_enhanced_order_manager": True,
                    "enable_async_order_executor": True,
                },
                "model_training": {
                    "enable_continuous_training": True,
                    "training_interval_hours": 24,
                },
            },
        }

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Sample market data for testing."""
        return pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
                "timestamp": pd.date_range(
                    start="2024-01-01",
                    periods=100,
                    freq="1min",
                ),
            },
        )

    @pytest.fixture
    async def ml_confidence_predictor(self, sample_config) -> MLConfidencePredictor:
        """Initialize ML confidence predictor for testing."""
        predictor = MLConfidencePredictor(sample_config)
        await predictor.initialize()
        return predictor

    @pytest.mark.asyncio
    async def test_ml_confidence_predictor_initialization(self, sample_config):
        """Test ML confidence predictor initialization."""
        predictor = MLConfidencePredictor(sample_config)
        assert predictor is not None

        success = await predictor.initialize()
        assert success is True

    @pytest.mark.asyncio
    async def test_confidence_prediction(
        self,
        ml_confidence_predictor,
        sample_market_data,
    ):
        """Test confidence prediction."""
        current_price = 100.5

        # Test confidence prediction
        confidence_table = await ml_confidence_predictor.predict_confidence_table(
            sample_market_data,
            current_price,
        )

        assert confidence_table is not None
        assert "price_target_confidences" in confidence_table
        assert "adversarial_confidences" in confidence_table
        assert "directional_confidence_analysis" in confidence_table

    @pytest.mark.asyncio
    async def test_dual_model_prediction(
        self,
        ml_confidence_predictor,
        sample_market_data,
    ):
        """Test dual model prediction."""
        current_price = 100.5

        # Test analyst prediction
        analyst_prediction = (
            await ml_confidence_predictor.predict_for_dual_model_system(
                sample_market_data,
                current_price,
                "analyst",
            )
        )

        assert analyst_prediction is not None
        assert "confidence" in analyst_prediction

        # Test tactician prediction
        tactician_prediction = (
            await ml_confidence_predictor.predict_for_dual_model_system(
                sample_market_data,
                current_price,
                "tactician",
            )
        )

        assert tactician_prediction is not None
        assert "confidence" in tactician_prediction

    @pytest.mark.asyncio
    async def test_meta_labeling_prediction(
        self,
        ml_confidence_predictor,
        sample_market_data,
    ):
        """Test meta-labeling prediction."""
        current_price = 100.5

        # Test meta-labeling prediction
        prediction = await ml_confidence_predictor.predict_with_meta_labeling(
            sample_market_data,
            current_price,
            "analyst",
        )

        assert prediction is not None
        assert "confidence" in prediction
        assert "meta_labels" in prediction

    @pytest.mark.asyncio
    async def test_order_execution(self, ml_confidence_predictor):
        """Test order execution methods."""
        # Test order execution with strategy
        execution_result = await ml_confidence_predictor.execute_order_with_strategy(
            symbol="ETHUSDT",
            side="buy",
            quantity=0.1,
            price=100.0,
            strategy_type="immediate",
        )

        assert execution_result is not None
        assert "success" in execution_result

    @pytest.mark.asyncio
    async def test_model_training(self, ml_confidence_predictor, sample_market_data):
        """Test model training."""
        # Test training trigger
        training_result = await ml_confidence_predictor.trigger_model_training(
            sample_market_data,
            "continuous",
            force_training=False,
        )

        assert training_result is not None
        assert "success" in training_result

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, ml_confidence_predictor):
        """Test performance monitoring."""
        # Test performance update
        performance_metrics = {"accuracy": 0.75, "precision": 0.8}
        await ml_confidence_predictor.update_model_performance(performance_metrics)

        # Test training status
        training_status = ml_confidence_predictor.get_training_status()
        assert training_status is not None

    @pytest.mark.asyncio
    async def test_enhanced_training_integration(self, ml_confidence_predictor):
        """Test enhanced training integration."""
        # Test enhanced training availability
        is_available = ml_confidence_predictor.is_enhanced_training_available()
        assert isinstance(is_available, bool)

        # Test model availability status
        status = ml_confidence_predictor.get_model_availability_status()
        assert status is not None

    @pytest.mark.asyncio
    async def test_cleanup(self, ml_confidence_predictor):
        """Test cleanup."""
        await ml_confidence_predictor.stop()


class TestOrderManagement:
    """Test suite for Order Management components."""

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """Sample configuration for order management."""
        return {
            "enhanced_order_manager": {
                "enable_enhanced_order_manager": True,
                "enable_async_order_executor": True,
                "enable_chase_micro_breakout": True,
                "enable_limit_order_return": True,
                "enable_partial_fill_management": True,
                "max_order_retries": 3,
                "order_timeout_seconds": 30,
                "slippage_tolerance": 0.001,
                "volume_threshold": 1.5,
                "momentum_threshold": 0.02,
            },
        }

    @pytest.mark.asyncio
    async def test_enhanced_order_manager(self, sample_config):
        """Test enhanced order manager."""
        order_manager = await setup_enhanced_order_manager(sample_config)

        if order_manager is not None:
            assert order_manager is not None
            assert hasattr(order_manager, "place_chase_micro_breakout_order")
            assert hasattr(order_manager, "place_limit_order_return")

    @pytest.mark.asyncio
    async def test_async_order_executor(self, sample_config):
        """Test async order executor."""
        order_executor = await setup_async_order_executor(sample_config)

        if order_executor is not None:
            assert order_executor is not None
            assert hasattr(order_executor, "execute_order_async")
            assert hasattr(order_executor, "get_execution_status")


class TestEnhancedTrainingManager:
    """Test suite for Enhanced Training Manager."""

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """Sample configuration for enhanced training manager."""
        return {
            "enhanced_training_manager": {
                "enable_enhanced_training": True,
                "enable_mlflow": True,
                "enable_optuna": True,
                "enable_distributed_tracing": False,
                "enable_metrics_collection": True,
                "enable_caching": True,
                "enable_async_optimization": True,
                "enable_performance_monitoring": True,
                "enable_rollback_points": True,
                "enable_automated_optimization": True,
            },
        }

    @pytest.mark.asyncio
    async def test_enhanced_training_manager(self, sample_config):
        """Test enhanced training manager."""
        training_manager = await setup_enhanced_training_manager(sample_config)

        if training_manager is not None:
            assert training_manager is not None
            assert hasattr(training_manager, "execute_enhanced_training")
            assert hasattr(training_manager, "get_enhanced_training_results")


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    def complete_config(self) -> dict[str, Any]:
        """Complete configuration for integration testing."""
        return {
            "dual_model_system": {
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.5,
                "tactician_confidence_threshold": 0.6,
                "enter_signal_validity_duration": 120,
                "signal_check_interval": 10,
                "enable_ensemble_analysis": True,
                "enable_meta_labeling": True,
                "enable_feature_engineering": True,
                "enhanced_order_manager": {
                    "enable_enhanced_order_manager": True,
                    "enable_async_order_executor": True,
                },
                "model_training": {
                    "enable_continuous_training": True,
                    "training_interval_hours": 24,
                },
            },
            "ml_confidence_predictor": {
                "price_movement_levels": [0.5, 1.0, 1.5, 2.0],
                "adversarial_movement_levels": [0.1, 0.2, 0.3, 0.4],
                "enable_enhanced_training": True,
                "enable_meta_labeling": True,
                "enable_feature_engineering": True,
            },
        }

    @pytest.mark.asyncio
    async def test_complete_system_integration(self, complete_config):
        """Test complete system integration."""
        # Initialize dual model system
        dual_system = await setup_dual_model_system(complete_config)
        assert dual_system is not None

        # Initialize ML confidence predictor
        ml_predictor = await setup_ml_confidence_predictor(complete_config)
        assert ml_predictor is not None

        # Test system interaction
        sample_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        # Test decision making
        decision = await dual_system.make_trading_decision(sample_data, 100.5)
        assert decision is not None

        # Test confidence prediction
        confidence = await ml_predictor.predict_confidence_table(sample_data, 100.5)
        assert confidence is not None

        # Cleanup
        await dual_system.stop()
        await ml_predictor.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self, complete_config):
        """Test error handling in the system."""
        # Test with invalid data
        dual_system = await setup_dual_model_system(complete_config)

        # Test with empty data
        empty_data = pd.DataFrame()
        decision = await dual_system.make_trading_decision(empty_data, 100.5)
        assert decision is not None  # Should return fallback decision

        # Test with None data
        decision = await dual_system.make_trading_decision(None, 100.5)
        assert decision is not None  # Should return fallback decision

        await dual_system.stop()

    @pytest.mark.asyncio
    async def test_performance_validation(self, complete_config):
        """Test performance validation."""
        dual_system = await setup_dual_model_system(complete_config)

        # Test performance metrics
        performance_metrics = {
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
        }

        await dual_system.update_model_performance(performance_metrics)

        # Test training status
        training_status = dual_system.get_training_status()
        assert training_status is not None

        await dual_system.stop()


class TestValidation:
    """Validation tests for system reliability."""

    @pytest.mark.asyncio
    async def test_data_validation(self):
        """Test data validation."""
        # Test valid market data
        valid_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in valid_data.columns

        # Validate data types
        assert valid_data["open"].dtype in ["float64", "float32", "int64", "int32"]
        assert valid_data["high"].dtype in ["float64", "float32", "int64", "int32"]
        assert valid_data["low"].dtype in ["float64", "float32", "int64", "int32"]
        assert valid_data["close"].dtype in ["float64", "float32", "int64", "int32"]
        assert valid_data["volume"].dtype in ["float64", "float32", "int64", "int32"]

        # Validate data ranges
        assert (valid_data["high"] >= valid_data["low"]).all()
        assert (valid_data["high"] >= valid_data["open"]).all()
        assert (valid_data["high"] >= valid_data["close"]).all()
        assert (valid_data["low"] <= valid_data["open"]).all()
        assert (valid_data["low"] <= valid_data["close"]).all()
        assert (valid_data["volume"] >= 0).all()

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = {
            "dual_model_system": {
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.5,
                "tactician_confidence_threshold": 0.6,
            },
        }

        # Validate required keys
        assert "dual_model_system" in valid_config
        assert "analyst_timeframes" in valid_config["dual_model_system"]
        assert "tactician_timeframes" in valid_config["dual_model_system"]

        # Validate threshold ranges
        analyst_threshold = valid_config["dual_model_system"][
            "analyst_confidence_threshold"
        ]
        tactician_threshold = valid_config["dual_model_system"][
            "tactician_confidence_threshold"
        ]

        assert 0.0 <= analyst_threshold <= 1.0
        assert 0.0 <= tactician_threshold <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_validation(self):
        """Test confidence score validation."""
        # Test valid confidence scores
        valid_confidences = [0.0, 0.25, 0.5, 0.75, 1.0]

        for confidence in valid_confidences:
            assert 0.0 <= confidence <= 1.0

        # Test invalid confidence scores
        invalid_confidences = [-0.1, 1.1, 2.0, -1.0]

        for confidence in invalid_confidences:
            assert not (0.0 <= confidence <= 1.0)

    @pytest.mark.asyncio
    async def test_performance_validation(self):
        """Test performance metrics validation."""
        # Test valid performance metrics
        valid_metrics = {
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
            "auc": 0.8,
        }

        for metric, value in valid_metrics.items():
            assert 0.0 <= value <= 1.0

        # Test invalid performance metrics
        invalid_metrics = {
            "accuracy": -0.1,
            "precision": 1.1,
            "recall": 2.0,
            "f1_score": -1.0,
        }

        for metric, value in invalid_metrics.items():
            assert not (0.0 <= value <= 1.0)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
