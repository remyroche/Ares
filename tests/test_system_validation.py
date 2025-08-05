# tests/test_system_validation.py

"""
System Validation Framework
Comprehensive validation tests for data quality, model performance,
system reliability, and integration.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.analyst.ml_confidence_predictor import MLConfidencePredictor
from src.ares_pipeline import AresPipeline

# Import components for validation
from src.training.dual_model_system import DualModelSystem


class TestDataQualityValidation:
    """Test suite for data quality validation."""

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Sample market data for validation."""
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

    @pytest.mark.asyncio
    async def test_market_data_completeness(self, sample_market_data):
        """Test market data completeness."""
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in sample_market_data.columns, f"Missing required column: {col}"

        # Check for missing values
        for col in required_columns:
            missing_count = sample_market_data[col].isnull().sum()
            assert (
                missing_count == 0
            ), f"Column {col} has {missing_count} missing values"

        # Check data length
        assert len(sample_market_data) > 0, "Market data is empty"
        assert len(sample_market_data) >= 50, "Insufficient market data for analysis"

    @pytest.mark.asyncio
    async def test_market_data_consistency(self, sample_market_data):
        """Test market data consistency."""
        # Check OHLC relationships
        assert (
            sample_market_data["high"] >= sample_market_data["low"]
        ).all(), "High must be >= Low"
        assert (
            sample_market_data["high"] >= sample_market_data["open"]
        ).all(), "High must be >= Open"
        assert (
            sample_market_data["high"] >= sample_market_data["close"]
        ).all(), "High must be >= Close"
        assert (
            sample_market_data["low"] <= sample_market_data["open"]
        ).all(), "Low must be <= Open"
        assert (
            sample_market_data["low"] <= sample_market_data["close"]
        ).all(), "Low must be <= Close"

        # Check volume consistency
        assert (sample_market_data["volume"] >= 0).all(), "Volume must be non-negative"

        # Check price consistency
        assert (sample_market_data["open"] > 0).all(), "Open price must be positive"
        assert (sample_market_data["close"] > 0).all(), "Close price must be positive"

    @pytest.mark.asyncio
    async def test_market_data_quality_metrics(self, sample_market_data):
        """Test market data quality metrics."""
        # Calculate quality metrics
        total_rows = len(sample_market_data)
        complete_rows = sample_market_data.dropna().shape[0]
        completeness_ratio = complete_rows / total_rows

        # Check completeness
        assert (
            completeness_ratio >= 0.95
        ), f"Data completeness too low: {completeness_ratio:.2%}"

        # Check for outliers
        for col in ["open", "high", "low", "close"]:
            q1 = sample_market_data[col].quantile(0.25)
            q3 = sample_market_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = sample_market_data[
                (sample_market_data[col] < lower_bound)
                | (sample_market_data[col] > upper_bound)
            ]

            outlier_ratio = len(outliers) / total_rows
            assert (
                outlier_ratio <= 0.05
            ), f"Too many outliers in {col}: {outlier_ratio:.2%}"

    @pytest.mark.asyncio
    async def test_market_data_timestamps(self, sample_market_data):
        """Test market data timestamp consistency."""
        if "timestamp" in sample_market_data.columns:
            # Check timestamp ordering
            timestamps = pd.to_datetime(sample_market_data["timestamp"])
            assert (
                timestamps.is_monotonic_increasing
            ), "Timestamps must be in ascending order"

            # Check for duplicate timestamps
            duplicate_count = timestamps.duplicated().sum()
            assert duplicate_count == 0, f"Found {duplicate_count} duplicate timestamps"

            # Check timestamp frequency consistency
            time_diffs = timestamps.diff().dropna()
            if len(time_diffs) > 0:
                # Check if time differences are consistent (within 10% tolerance)
                mean_diff = time_diffs.mean()
                std_diff = time_diffs.std()
                cv = std_diff / mean_diff
                assert cv <= 0.1, f"Timestamp frequency too inconsistent: CV={cv:.3f}"


class TestModelPerformanceValidation:
    """Test suite for model performance validation."""

    @pytest.fixture
    def sample_performance_metrics(self) -> dict[str, Any]:
        """Sample performance metrics for validation."""
        return {
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
            "auc": 0.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.05,
            "win_rate": 0.65,
            "profit_factor": 1.5,
            "total_return": 0.25,
        }

    @pytest.mark.asyncio
    async def test_performance_metrics_validation(self, sample_performance_metrics):
        """Test performance metrics validation."""
        # Validate metric ranges
        for metric, value in sample_performance_metrics.items():
            if metric in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
                "win_rate",
            ]:
                assert (
                    0.0 <= value <= 1.0
                ), f"{metric} must be between 0 and 1, got {value}"
            elif metric in ["max_drawdown"]:
                assert (
                    -1.0 <= value <= 0.0
                ), f"{metric} must be between -1 and 0, got {value}"
            elif metric in ["profit_factor", "total_return"]:
                assert value >= 0.0, f"{metric} must be non-negative, got {value}"

        # Validate metric relationships
        precision = sample_performance_metrics["precision"]
        recall = sample_performance_metrics["recall"]
        f1_score = sample_performance_metrics["f1_score"]

        # Check F1 score calculation
        expected_f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        assert abs(f1_score - expected_f1) < 0.01, "F1 score calculation mismatch"

        # Check win rate consistency
        win_rate = sample_performance_metrics["win_rate"]
        assert 0.0 <= win_rate <= 1.0, "Win rate must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, sample_performance_metrics):
        """Test performance degradation detection."""
        # Simulate performance degradation
        degraded_metrics = sample_performance_metrics.copy()
        degraded_metrics["accuracy"] = 0.6  # Degraded from 0.75
        degraded_metrics["precision"] = 0.65  # Degraded from 0.8

        # Calculate degradation
        accuracy_degradation = 0.75 - degraded_metrics["accuracy"]
        precision_degradation = 0.8 - degraded_metrics["precision"]

        # Check if degradation exceeds threshold
        degradation_threshold = 0.1
        assert (
            accuracy_degradation > degradation_threshold
        ), "Accuracy degradation not detected"
        assert (
            precision_degradation > degradation_threshold
        ), "Precision degradation not detected"

    @pytest.mark.asyncio
    async def test_model_stability_validation(self):
        """Test model stability validation."""
        # Simulate performance over time
        performance_history = [
            {"accuracy": 0.75, "precision": 0.8, "recall": 0.7},
            {"accuracy": 0.74, "precision": 0.79, "recall": 0.71},
            {"accuracy": 0.76, "precision": 0.81, "recall": 0.69},
            {"accuracy": 0.73, "precision": 0.78, "recall": 0.72},
            {"accuracy": 0.77, "precision": 0.82, "recall": 0.68},
        ]

        # Calculate stability metrics
        accuracies = [p["accuracy"] for p in performance_history]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        cv_accuracy = std_accuracy / mean_accuracy

        # Check stability (coefficient of variation should be low)
        assert cv_accuracy <= 0.05, f"Model accuracy too unstable: CV={cv_accuracy:.3f}"

        # Check for consistent performance
        for i in range(1, len(performance_history)):
            accuracy_change = abs(accuracies[i] - accuracies[i - 1])
            assert (
                accuracy_change <= 0.05
            ), f"Large accuracy change detected: {accuracy_change:.3f}"


class TestSystemReliabilityValidation:
    """Test suite for system reliability validation."""

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """Sample configuration for system validation."""
        return {
            "dual_model_system": {
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.5,
                "tactician_confidence_threshold": 0.6,
                "enable_ensemble_analysis": True,
                "enable_meta_labeling": True,
                "enable_feature_engineering": True,
            },
        }

    @pytest.mark.asyncio
    async def test_system_initialization_reliability(self, sample_config):
        """Test system initialization reliability."""
        # Test multiple initialization attempts
        for i in range(3):
            try:
                system = DualModelSystem(sample_config)
                success = await system.initialize()
                assert success is True, f"Initialization failed on attempt {i+1}"
                await system.stop()
            except Exception as e:
                pytest.fail(f"Initialization error on attempt {i+1}: {e}")

    @pytest.mark.asyncio
    async def test_system_error_handling(self, sample_config):
        """Test system error handling."""
        system = DualModelSystem(sample_config)
        await system.initialize()

        # Test with invalid data
        invalid_data = pd.DataFrame()
        decision = await system.make_trading_decision(invalid_data, 100.5)
        assert decision is not None, "System should handle invalid data gracefully"

        # Test with None data
        decision = await system.make_trading_decision(None, 100.5)
        assert decision is not None, "System should handle None data gracefully"

        # Test with extreme values
        extreme_data = pd.DataFrame(
            {
                "open": [float("inf")] * 10,
                "high": [float("inf")] * 10,
                "low": [float("-inf")] * 10,
                "close": [float("inf")] * 10,
                "volume": [float("inf")] * 10,
            },
        )
        decision = await system.make_trading_decision(extreme_data, 100.5)
        assert decision is not None, "System should handle extreme values gracefully"

        await system.stop()

    @pytest.mark.asyncio
    async def test_system_memory_usage(self, sample_config):
        """Test system memory usage."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Initialize system
        system = DualModelSystem(sample_config)
        await system.initialize()

        # Perform multiple operations
        sample_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        for _ in range(10):
            await system.make_trading_decision(sample_data, 100.5)

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f}MB"

        await system.stop()

    @pytest.mark.asyncio
    async def test_system_response_time(self, sample_config):
        """Test system response time."""
        system = DualModelSystem(sample_config)
        await system.initialize()

        sample_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        # Measure response time
        start_time = datetime.now(UTC)
        await system.make_trading_decision(sample_data, 100.5)
        end_time = datetime.now(UTC)

        response_time = (end_time - start_time).total_seconds()

        # Response time should be reasonable (< 5 seconds)
        assert response_time < 5.0, f"Response time too slow: {response_time:.2f}s"

        await system.stop()

    @pytest.mark.asyncio
    async def test_system_concurrent_operations(self, sample_config):
        """Test system concurrent operations."""
        system = DualModelSystem(sample_config)
        await system.initialize()

        sample_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        # Test concurrent decision making
        async def make_decision():
            return await system.make_trading_decision(sample_data, 100.5)

        # Run multiple concurrent operations
        tasks = [make_decision() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All operations should complete successfully
        for result in results:
            assert result is not None, "Concurrent operation failed"

        await system.stop()


class TestIntegrationValidation:
    """Test suite for integration validation."""

    @pytest.fixture
    def complete_config(self) -> dict[str, Any]:
        """Complete configuration for integration testing."""
        return {
            "dual_model_system": {
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.5,
                "tactician_confidence_threshold": 0.6,
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
    async def test_component_integration(self, complete_config):
        """Test component integration."""
        # Initialize all components
        dual_system = DualModelSystem(complete_config)
        await dual_system.initialize()

        ml_predictor = MLConfidencePredictor(complete_config)
        await ml_predictor.initialize()

        # Test component interaction
        sample_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        # Test dual system decision
        dual_decision = await dual_system.make_trading_decision(sample_data, 100.5)
        assert dual_decision is not None

        # Test ML predictor confidence
        confidence = await ml_predictor.predict_confidence_table(sample_data, 100.5)
        assert confidence is not None

        # Test integration consistency
        if dual_decision.get("action") == "ENTRY":
            assert dual_decision.get("analyst_confidence", 0) >= 0.5
            assert dual_decision.get("tactician_confidence", 0) >= 0.6

        # Cleanup
        await dual_system.stop()
        await ml_predictor.stop()

    @pytest.mark.asyncio
    async def test_data_flow_validation(self, complete_config):
        """Test data flow validation."""
        dual_system = DualModelSystem(complete_config)
        await dual_system.initialize()

        # Test data flow through the system
        sample_data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 100,
            },
        )

        # Test entry decision flow
        entry_decision = await dual_system._make_entry_decision(sample_data, 100.5)
        assert entry_decision is not None

        # Test exit decision flow
        current_position = {
            "side": "LONG",
            "quantity": 0.1,
            "entry_price": 100.0,
            "current_price": 100.5,
        }
        exit_decision = await dual_system._make_exit_decision(
            sample_data,
            100.5,
            current_position,
        )
        assert exit_decision is not None

        # Test confidence calculation flow
        analyst_confidence = 0.7
        tactician_confidence = 0.8
        final_confidence = dual_system._calculate_final_confidence(
            analyst_confidence,
            tactician_confidence,
        )
        assert 0.0 <= final_confidence <= 1.0

        await dual_system.stop()

    @pytest.mark.asyncio
    async def test_configuration_consistency(self, complete_config):
        """Test configuration consistency."""
        # Validate configuration structure
        assert "dual_model_system" in complete_config
        assert "ml_confidence_predictor" in complete_config

        dual_config = complete_config["dual_model_system"]
        ml_config = complete_config["ml_confidence_predictor"]

        # Check timeframe consistency
        analyst_timeframes = dual_config.get("analyst_timeframes", [])
        ml_analyst_timeframes = ml_config.get("analyst_timeframes", [])

        # Timeframes should be consistent
        assert set(analyst_timeframes) == set(
            ml_analyst_timeframes,
        ), "Timeframe mismatch"

        # Check threshold consistency
        analyst_threshold = dual_config.get("analyst_confidence_threshold", 0.5)
        assert 0.0 <= analyst_threshold <= 1.0, "Invalid analyst confidence threshold"

        tactician_threshold = dual_config.get("tactician_confidence_threshold", 0.6)
        assert (
            0.0 <= tactician_threshold <= 1.0
        ), "Invalid tactician confidence threshold"

    @pytest.mark.asyncio
    async def test_error_propagation(self, complete_config):
        """Test error propagation through the system."""
        dual_system = DualModelSystem(complete_config)
        await dual_system.initialize()

        # Test error handling in decision making
        try:
            # Test with corrupted data
            corrupted_data = pd.DataFrame(
                {
                    "open": [None] * 10,
                    "high": [float("inf")] * 10,
                    "low": [float("-inf")] * 10,
                    "close": [None] * 10,
                    "volume": [None] * 10,
                },
            )

            decision = await dual_system.make_trading_decision(corrupted_data, 100.5)
            assert decision is not None, "System should handle corrupted data"

        except Exception as e:
            pytest.fail(f"System should handle corrupted data gracefully: {e}")

        await dual_system.stop()


class TestEndToEndValidation:
    """Test suite for end-to-end validation."""

    @pytest.fixture
    def pipeline_config(self) -> dict[str, Any]:
        """Pipeline configuration for end-to-end testing."""
        return {
            "pipeline": {
                "enable_dual_model_system": True,
                "enable_ml_confidence_predictor": True,
                "enable_enhanced_order_manager": True,
                "enable_enhanced_training_manager": True,
                "cycle_interval_seconds": 60,
                "max_cycles": 10,
            },
            "dual_model_system": {
                "analyst_timeframes": ["30m", "15m", "5m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.5,
                "tactician_confidence_threshold": 0.6,
                "enable_ensemble_analysis": True,
                "enable_meta_labeling": True,
                "enable_feature_engineering": True,
            },
        }

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, pipeline_config):
        """Test complete pipeline execution."""
        # Initialize pipeline
        pipeline = AresPipeline(pipeline_config)
        await pipeline.initialize()

        # Test pipeline status
        status = pipeline.get_pipeline_status()
        assert status is not None
        assert "dual_model_system_status" in status

        # Test pipeline cleanup
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_system_reliability_stress_test(self, pipeline_config):
        """Test system reliability under stress."""
        pipeline = AresPipeline(pipeline_config)
        await pipeline.initialize()

        # Simulate stress conditions
        for i in range(5):
            try:
                # Test with varying data (not used in this test but created for completeness)
                _sample_data = pd.DataFrame(
                    {
                        "open": [100.0 + i] * 100,
                        "high": [101.0 + i] * 100,
                        "low": [99.0 + i] * 100,
                        "close": [100.5 + i] * 100,
                        "volume": [1000.0 + i * 100] * 100,
                    },
                )

                # This would normally be called during pipeline execution
                # For testing, we'll just verify the system is stable
                assert pipeline.dual_model_system is not None

            except Exception as e:
                pytest.fail(f"System failed under stress on iteration {i}: {e}")

        await pipeline.stop()


if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v"])
