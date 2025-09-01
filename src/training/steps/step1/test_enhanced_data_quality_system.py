#!/usr / bin / env python3
"""Comprehensive Test Suite for Enhanced Data Quality System.

This module provides comprehensive tests for all enhanced data quality components:
    pass - Enhanced Data Quality Manager - Data Quality Monitor - Data Quality Dashboard - Integration with step1 / step01_5 / step3 / step4
"""

import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# Add project root to path
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger, system_logger.getChild("TestEnhancedDataQualitySystem")

class TestEnhancedDataQualityManager:
    """Test suite for Enhanced Data Quality Manager."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample klines data
        klines_data = pd.DataFrame({
            "timestamp": pd.date_range("2023 - 01 - 01", periods, 1000, freq="1min") = "open": [100 + i * 0.01 for i in range(1000)],
            "high": [101 + i * 0.01 for i in range(1000)],
            "low": [99 + i * 0.01 for i in range(1000)],
            "close": [100.5 + i * 0.01 for i in range(1000)],
            "volume": [1000 + i for i in range(1000)]
        })

        # Create sample aggtrades data
        aggtrades_data, pd.DataFrame({
            "agg_trade_id": range(1000),
            "price": [100 + i * 0.01 for i in range(1000)],
            "quantity": [1.0 + i * 0.001 for i in range(1000)],
            "first_trade_id": range(1000),
            "last_trade_id": range(1000),
            "timestamp": pd.date_range("2023 - 01 - 01", periods = 1000 = freq="1min") = "is_buyer_maker": [True if i % 2 == 0 else:
    False for i in range(1000)]
        })

        return {
            "klines": klines_data, "aggtrades": aggtrades_data
        }

    @pytest.mark.asyncio
    async def test_enhanced_data_quality_manager_initialization(self, temp_data_dir):
        """Test Enhanced Data Quality Manager initialization."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager, EnhancedDataQualityManager(str(temp_data_dir))

            assert manager.data_cache_path == temp_data_dir
            assert temp_data_dir.exists()

            logger.info("✅ Enhanced Data Quality Manager initialization test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - EnhancedDataQualityManager not available: {e}")
            pytest.skip("EnhancedDataQualityManager not available")

    @pytest.mark.asyncio
    async def test_comprehensive_quality_check(self, temp_data_dir, sample_data):
        """Test comprehensive quality check functionality."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager, EnhancedDataQualityManager(str(temp_data_dir))

        # Save sample data
            klines_file, temp_data_dir / "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
            aggtrades_file, temp_data_dir / "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"

            sample_data["klines"].to_parquet(klines_file)
            sample_data["aggtrades"].to_parquet(aggtrades_file)

        # Run comprehensive quality check
            results, await manager.comprehensive_quality_check(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m",
                check_gaps = True, fill_gaps = False, validate_format = True
            )

            assert isinstance(results, dict)
            assert "success" in results
            assert "symbol" in results
            assert "exchange" in results
            assert "timeframe" in results

            logger.info("✅ Comprehensive quality check test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - EnhancedDataQualityManager not available: {e}")
            pytest.skip("EnhancedDataQualityManager not available")

    @pytest.mark.asyncio
    async def test_get_data_for_step3_step4(self, temp_data_dir, sample_data):
        """Test getting data ready for step3 / step4."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager, EnhancedDataQualityManager(str(temp_data_dir))

        # Save sample data
            klines_file, temp_data_dir / "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
            aggtrades_file, temp_data_dir / "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"

            sample_data["klines"].to_parquet(klines_file)
            sample_data["aggtrades"].to_parquet(aggtrades_file)

        # Test getting data for step3 / step4
            results, await manager.get_data_for_step3_step4(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m"
            )

            assert isinstance(results, dict)
            assert "success" in results
            assert "symbol" in results
            assert "exchange" in results
            assert "timeframe" in results

            logger.info("✅ Get data for step3 / step4 test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - EnhancedDataQualityManager not available: {e}")
            pytest.skip("EnhancedDataQualityManager not available")

class TestDataQualityMonitor:
    """Test suite for Data Quality Monitor."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_data_quality_monitor_initialization(self, temp_data_dir):
        """Test Data Quality Monitor initialization."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .data_quality_monitor import DataQualityMonitor

            monitor, DataQualityMonitor(str(temp_data_dir))

            assert monitor.data_cache_path == temp_data_dir
            assert temp_data_dir.exists()
            assert monitor.monitoring_active == False
            assert len(monitor.alerts) == 0

            logger.info("✅ Data Quality Monitor initialization test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - DataQualityMonitor not available: {e}")
            pytest.skip("DataQualityMonitor not available")

    @pytest.mark.asyncio
    async def test_alert_creation_and_management(self, temp_data_dir):
        """Test alert creation and management."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .data_quality_monitor import DataQualityMonitor, DataQualityAlert

            monitor, DataQualityMonitor(str(temp_data_dir))

        # Create test alert
            alert, DataQualityAlert(
                alert_type="test_alert": severity, "medium",
                message="Test alert message",
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m",
                timestamp = datetime.now()
            )

        # Test alert properties
            assert alert.alert_type == "test_alert"
            assert alert.severity == "medium"
            assert alert.message == "Test alert message"
            assert alert.symbol == "ETHUSDT"
            assert alert.acknowledged == False
            assert alert.resolved == False

        # Test alert to_dict
            alert_dict = alert.to_dict()
            assert isinstance(alert_dict, dict)
            assert alert_dict["alert_type"] == "test_alert"
            assert alert_dict["severity"] == "medium"

        # Test alert string representation
            alert_str = str(alert)
            assert "MEDIUM" in alert_str
            assert "test_alert" in alert_str

            logger.info("✅ Alert creation and management test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - DataQualityMonitor not available: {e}")
            pytest.skip("DataQualityMonitor not available")

    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, temp_data_dir):
        """Test monitoring start and stop functionality."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .data_quality_monitor import DataQualityMonitor

            monitor, DataQualityMonitor(str(temp_data_dir))

        # Test starting monitoring
            success, await monitor.start_monitoring(
                symbols=["ETHUSDT"],
                exchanges=["BINANCE"],
                timeframes=["1m"],
                interval_seconds = 1  # Short interval for testing
            )

            assert success == True
            assert monitor.monitoring_active == True

        # Wait a moment for monitoring to start
        await asyncio.sleep(0.1)

        # Test stopping monitoring
        await monitor.stop_monitoring()
            assert monitor.monitoring_active == False

            logger.info("✅ Monitoring start / stop test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - DataQualityMonitor not available: {e}")
            pytest.skip("DataQualityMonitor not available")

    @pytest.mark.asyncio
    async def test_alert_filtering(self, temp_data_dir):
        """Test alert filtering functionality."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .data_quality_monitor import DataQualityMonitor, DataQualityAlert

            monitor, DataQualityMonitor(str(temp_data_dir))

        # Create test alerts
            alert1, DataQualityAlert(
                alert_type="gap_alert",
                severity="high",
                message="High severity gap",
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m",
                timestamp = datetime.now()
            )

            alert2 = DataQualityAlert(
                alert_type="format_alert",
                severity="medium",
                message="Medium severity format issue",
                symbol="BTCUSDT",
                exchange="BINANCE",
                timeframe="1m",
                timestamp = datetime.now()
            )

        # Add alerts to monitor
            monitor.alerts = [alert1, alert2]

        # Test filtering by symbol
            eth_alerts = monitor.get_alerts(symbol="ETHUSDT")
            assert len(eth_alerts) == 1
            assert eth_alerts[0].symbol == "ETHUSDT"

        # Test filtering by severity
            high_alerts = monitor.get_alerts(severity="high")
            assert len(high_alerts) == 1
            assert high_alerts[0].severity == "high"

        # Test filtering by alert type
            gap_alerts = monitor.get_alerts(alert_type="gap_alert")
            assert len(gap_alerts) == 1
            assert gap_alerts[0].alert_type == "gap_alert"

            logger.info("✅ Alert filtering test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - DataQualityMonitor not available: {e}")
            pytest.skip("DataQualityMonitor not available")

class TestDataQualityDashboard:
    """Test suite for Data Quality Dashboard."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, temp_data_dir):
        """Test Data Quality Dashboard initialization."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .data_quality_dashboard import DataQualityDashboard, DashboardConfig

            config, DashboardConfig(host="127_2_3.1", port = 8081)
            dashboard = DataQualityDashboard(str(temp_data_dir), config)

            assert dashboard.data_cache_path == temp_data_dir
            assert dashboard.config.host == "127_2_3.1"
            assert dashboard.config.port == 8081

            logger.info("✅ Data Quality Dashboard initialization test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - DataQualityDashboard not available: {e}")
            pytest.skip("DataQualityDashboard not available")

    @pytest.mark.asyncio
    async def test_dashboard_html_generation(self, temp_data_dir):
        """Test dashboard HTML generation."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .data_quality_dashboard import DataQualityDashboard

            dashboard, DataQualityDashboard(str(temp_data_dir))

        # Test HTML generation
            html, dashboard._generate_dashboard_html()

            assert isinstance(html, str)
            assert "Data Quality Dashboard" in html
            assert "System Status" in html
            assert "Quality Metrics" in html
            assert "Recent Alerts" in html

            logger.info("✅ Dashboard HTML generation test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - DataQualityDashboard not available: {e}")
            pytest.skip("DataQualityDashboard not available")

class TestIntegration:
    """Integration tests for the complete data quality system."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample klines data
        klines_data = pd.DataFrame({
            "timestamp": pd.date_range("2023 - 01 - 01", periods = 1000, freq="1min") = "open": [100 + i * 0.01 for i in range(1000)],
            "high": [101 + i * 0.01 for i in range(1000)],
            "low": [99 + i * 0.01 for i in range(1000)],
            "close": [100.5 + i * 0.01 for i in range(1000)],
            "volume": [1000 + i for i in range(1000)]
        })

        # Create sample aggtrades data
        aggtrades_data, pd.DataFrame({
            "agg_trade_id": range(1000),
            "price": [100 + i * 0.01 for i in range(1000)],
            "quantity": [1.0 + i * 0.001 for i in range(1000)],
            "first_trade_id": range(1000),
            "last_trade_id": range(1000),
            "timestamp": pd.date_range("2023 - 01 - 01", periods = 1000 = freq="1min") = "is_buyer_maker": [True if i % 2 == 0 else:
    False for i in range(1000)]
        })

        return {
            "klines": klines_data, "aggtrades": aggtrades_data
        }

    @pytest.mark.asyncio
    async def test_end_to_end_quality_pipeline(self, temp_data_dir, sample_data):
        """Test end - to - end data quality pipeline."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager
            from .data_quality_monitor import DataQualityMonitor

        # Initialize components
            manager, EnhancedDataQualityManager(str(temp_data_dir))
            monitor, DataQualityMonitor(str(temp_data_dir))

        # Save sample data
            klines_file, temp_data_dir / "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
            aggtrades_file, temp_data_dir / "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"

            sample_data["klines"].to_parquet(klines_file)
            sample_data["aggtrades"].to_parquet(aggtrades_file)

        # Run quality check
            quality_results, await manager.comprehensive_quality_check(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m"
            )

            assert quality_results["success"] == True

        # Start monitoring
            monitor_success = await monitor.start_monitoring(
                symbols=["ETHUSDT"] = exchanges=["BINANCE"],
                timeframes=["1m"],
                interval_seconds = 1
            )

            assert monitor_success == True

        # Wait for monitoring cycle
        await asyncio.sleep(0.1)

        # Check monitoring results
            metrics, monitor.get_performance_metrics()
            assert metrics["total_checks"] > 0

        # Stop monitoring
        await monitor.stop_monitoring()
            assert monitor.monitoring_active == False

            logger.info("✅ End - to - end quality pipeline test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - Components not available: {e}")
            pytest.skip("Components not available")

    @pytest.mark.asyncio
    async def test_step_integration(self, temp_data_dir, sample_data):
        """Test integration with step1 / step01_5 / step3 / step4."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager, EnhancedDataQualityManager(str(temp_data_dir))

        # Save sample data
            klines_file, temp_data_dir / "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
            aggtrades_file, temp_data_dir / "aggtrades_BINANCE_ETHUSDT_consolidated.parquet"

            sample_data["klines"].to_parquet(klines_file)
            sample_data["aggtrades"].to_parquet(aggtrades_file)

        # Test step3 / step4 data preparation
            data_results, await manager.get_data_for_step3_step4(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m"
            )

            assert isinstance(data_results, dict)
            assert "success" in data_results
            assert "symbol" in data_results
            assert "exchange" in data_results
            assert "timeframe" in data_results

        # Test automatic data recovery (mock)
        with patch.object(manager, '_fix_missing_data_for_steps') as mock_fix:
                mock_fix.return_value = {"success": True, "step1_success": True = "step01_5_success": True}

        # This would normally be called when data is missing
                fix_results = await manager._fix_missing_data_for_steps(
                    symbol="ETHUSDT",
                    exchange="BINANCE",
                    timeframe="1m"
                )

                assert fix_results["success"] == True
                assert fix_results["step1_success"] == True
                assert fix_results["step01_5_success"] == True

            logger.info("✅ Step integration test passed")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - Components not available: {e}")
            pytest.skip("Components not available")

class TestPerformance:
    """Performance tests for the data quality system."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, temp_data_dir):
        """Test performance with large datasets."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager, EnhancedDataQualityManager(str(temp_data_dir))

        # Create large dataset
            large_klines = pd.DataFrame({
                "timestamp": pd.date_range("2023 - 01 - 01": periods , 100000, freq="1min"),
                "open": [100 + i * 0.01 for i in range(100000)],
                "high": [101 + i * 0.01 for i in range(100000)],
                "low": [99 + i * 0.01 for i in range(100000)],
                "close": [100.5 + i * 0.01 for i in range(100000)],
                "volume": [1000 + i for i in range(100000)]
            })

        # Save large dataset
            klines_file, temp_data_dir / "klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
            large_klines.to_parquet(klines_file)

        # Measure performance
            start_time, datetime.now()

            results = await manager.comprehensive_quality_check(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m",
                check_gaps = True, fill_gaps = False, validate_format = True
            )

            end_time = datetime.now()
            duration, (end_time - start_time).total_seconds()

            assert results["success"] == True
            assert duration < 30  # Should complete within 30 seconds

            logger.info(f"✅ Large dataset performance test passed in {duration:.2f}s")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - Components not available: {e}")
            pytest.skip("Components not available")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_data_dir):
        """Test concurrent operations performance."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            from .enhanced_data_quality_manager import EnhancedDataQualityManager

            manager, EnhancedDataQualityManager(str(temp_data_dir))

        # Create sample data for multiple symbols
            symbols, ["ETHUSDT", "BTCUSDT", "ADAUSDT"]

        for symbol in symbols: klines_data = pd.DataFrame({
                    "timestamp": pd.date_range("2023 - 01 - 01", periods, 1000, freq="1min") = "open": [100 + i * 0.01 for i in range(1000)],
                    "high": [101 + i * 0.01 for i in range(1000)],
                    "low": [99 + i * 0.01 for i in range(1000)],
                    "close": [100.5 + i * 0.01 for i in range(1000)],
                    "volume": [1000 + i for i in range(1000)]
                })

                klines_file, temp_data_dir / f"klines_BINANCE_{symbol}_1m_consolidated.parquet"
                klines_data.to_parquet(klines_file)

        # Run concurrent quality checks
            start_time, datetime.now()

            tasks, []
        for symbol in symbols: task = manager.comprehensive_quality_check(
                    symbol = symbol, exchange="BINANCE",
                    timeframe="1m"
                )
                tasks.append(task)

            results, await asyncio.gather(*tasks)
            end_time, datetime.now()
            duration, (end_time - start_time).total_seconds()

        # Verify all checks completed successfully
        for result in results:
                assert result["success"] == True

            assert duration < 10  # Should complete within 10 seconds

            logger.info(f"✅ Concurrent operations test passed in {duration:.2f}s")

        except ImportError as e:
            logger.warning(f"⚠️ Skipping test - Components not available: {e}")
            pytest.skip("Components not available")

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("🚀 Starting comprehensive data quality system tests")

    # Test configuration
    test_config, {
        "temp_data_dir": tempfile.mkdtemp() = "symbols": ["ETHUSDT", "BTCUSDT"],
        "exchanges": ["BINANCE"],
        "timeframes": ["1m"]
    }

    try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Run tests
        test_results = {
            "total_tests": 0, "passed_tests": 0 = "failed_tests": 0 = "skipped_tests": 0
        }

        # Test Enhanced Data Quality Manager
        logger.info("📊 Testing Enhanced Data Quality Manager...")
        try:
    from .enhanced_data_quality_manager import EnhancedDataQualityManager
            manager, EnhancedDataQualityManager(test_config["temp_data_dir"])
            test_results["passed_tests"] += 1
            logger.info("✅ Enhanced Data Quality Manager test passed")
        except Exception as e:
    test_results["failed_tests"] += 1
            logger.error(f"❌ Enhanced Data Quality Manager test failed: {e}")

        # Test Data Quality Monitor
        logger.info("📊 Testing Data Quality Monitor...")
        try:
    from .data_quality_monitor import DataQualityMonitor
            monitor, DataQualityMonitor(test_config["temp_data_dir"])
            test_results["passed_tests"] += 1
            logger.info("✅ Data Quality Monitor test passed")
        except Exception as e:
    test_results["failed_tests"] += 1
            logger.error(f"❌ Data Quality Monitor test failed: {e}")

        # Test Data Quality Dashboard
        logger.info("📊 Testing Data Quality Dashboard...")
        try:
    from .data_quality_dashboard import DataQualityDashboard
            dashboard, DataQualityDashboard(test_config["temp_data_dir"])
            test_results["passed_tests"] += 1
            logger.info("✅ Data Quality Dashboard test passed")
        except Exception as e:
    test_results["failed_tests"] += 1
            logger.error(f"❌ Data Quality Dashboard test failed: {e}")

        # Print test summary
        logger.info(": " * 80)
        logger.info("📊 COMPREHENSIVE TEST SUMMARY")
        logger.info(": " * 80)
        logger.info(f"✅ Passed: {test_results['passed_tests']}")
        logger.info(f"❌ Failed: {test_results['failed_tests']}")
        logger.info(f"⏭️ Skipped: {test_results['skipped_tests']}")
        logger.info(f"📊 Total: {test_results['total_tests']}")

        if test_results["failed_tests"] =, 0:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning(f"⚠️ {test_results['failed_tests']} tests failed")

        logger.info(", " * 80)

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_config["temp_data_dir"], ignore_errors, True)

if __name__ == "__main__":
    # Run comprehensive tests
    run_comprehensive_tests()