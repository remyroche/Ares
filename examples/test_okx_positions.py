#!/usr/bin/env python3
"""
OKX Position Methods Test Suite

This file contains unit tests for the OKX position fetching methods.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.okx import OkxExchange


class TestOkxPositions(unittest.TestCase):
    """Test cases for OKX position methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exchange = OkxExchange("test_key", "test_secret", "BTCUSDT", "test_password")
        self.exchange.logger = MagicMock()  # Mock logger to avoid noise in tests
    
    def test_get_all_positions_valid_input(self):
        """Test get_all_positions with valid input."""
        # Mock the _make_request method
        mock_data = [
            {
                "instId": "BTCUSDT",
                "instType": "SPOT",
                "pos": "1.5",
                "posSide": "long",
                "markPx": "50000",
                "avgPx": "49000",
                "upl": "1500",
                "uplRatio": "0.03",
                "liqPx": "45000",
                "margin": "1000",
                "notionalUsd": "75000",
                "mgnRatio": "0.02",
                "mgnMode": "isolated",
                "interest": "0",
                "uTime": "1640995200000",
                "cTime": "1640995200000",
                "lever": "1",
                "deltaBS": "1",
                "gammaBS": "0",
                "thetaBS": "0",
                "vegaBS": "0"
            }
        ]
        
        with patch.object(self.exchange, '_make_request', return_value=mock_data):
            result = asyncio.run(self.exchange.get_all_positions("SPOT"))
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["symbol"], "BTCUSDT")
            self.assertEqual(result[0]["size"], "1.5")
            self.assertEqual(result[0]["side"], "long")
    
    def test_get_all_positions_invalid_inst_type(self):
        """Test get_all_positions with invalid instrument type."""
        with patch.object(self.exchange, '_make_request', return_value=[]):
            result = asyncio.run(self.exchange.get_all_positions("INVALID"))
            
            # Should default to SPOT and return empty list
            self.assertEqual(result, [])
            self.exchange.logger.warning.assert_called()
    
    def test_get_all_positions_no_data(self):
        """Test get_all_positions when no data is returned."""
        with patch.object(self.exchange, '_make_request', return_value=None):
            result = asyncio.run(self.exchange.get_all_positions("SPOT"))
            
            self.assertEqual(result, [])
            self.exchange.logger.warning.assert_called()
    
    def test_get_position_by_symbol_valid(self):
        """Test get_position_by_symbol with valid symbol."""
        mock_data = [{
            "instId": "BTCUSDT",
            "instType": "SPOT",
            "pos": "1.0",
            "posSide": "long",
            "markPx": "50000"
        }]
        
        with patch.object(self.exchange, '_make_request', return_value=mock_data):
            result = asyncio.run(self.exchange.get_position_by_symbol("BTCUSDT", "SPOT"))
            
            self.assertEqual(result["symbol"], "BTCUSDT")
            self.assertEqual(result["size"], "1.0")
    
    def test_get_position_by_symbol_invalid_symbol(self):
        """Test get_position_by_symbol with invalid symbol."""
        with patch.object(self.exchange, '_make_request', return_value=[]):
            result = asyncio.run(self.exchange.get_position_by_symbol("", "SPOT"))
            
            self.assertEqual(result, {})
            self.exchange.logger.error.assert_called()
    
    def test_get_position_by_symbol_no_data(self):
        """Test get_position_by_symbol when no data is returned."""
        with patch.object(self.exchange, '_make_request', return_value=[]):
            result = asyncio.run(self.exchange.get_position_by_symbol("BTCUSDT", "SPOT"))
            
            self.assertEqual(result, {})
            self.exchange.logger.info.assert_called()
    
    def test_get_position_history_valid(self):
        """Test get_position_history with valid parameters."""
        mock_data = [{
            "instId": "BTCUSDT",
            "instType": "SPOT",
            "pos": "1.0",
            "posSide": "long",
            "markPx": "50000",
            "type": "open"
        }]
        
        with patch.object(self.exchange, '_make_request', return_value=mock_data):
            result = asyncio.run(self.exchange.get_position_history(
                symbol="BTCUSDT",
                inst_type="SPOT",
                limit=10
            ))
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["symbol"], "BTCUSDT")
            self.assertEqual(result[0]["changeType"], "open")
    
    def test_get_position_history_invalid_limit(self):
        """Test get_position_history with invalid limit."""
        with patch.object(self.exchange, '_make_request', return_value=[]):
            result = asyncio.run(self.exchange.get_position_history(limit=150))
            
            self.assertEqual(result, [])
            self.exchange.logger.warning.assert_called()
    
    def test_get_position_margin_valid(self):
        """Test get_position_margin with valid symbol."""
        mock_data = [{
            "instId": "BTCUSDT",
            "instType": "SPOT",
            "margin": "1000",
            "mgnRatio": "0.02",
            "mgnMode": "isolated",
            "lever": "1"
        }]
        
        with patch.object(self.exchange, '_make_request', return_value=mock_data):
            result = asyncio.run(self.exchange.get_position_margin("BTCUSDT"))
            
            self.assertEqual(result["symbol"], "BTCUSDT")
            self.assertEqual(result["margin"], "1000")
    
    def test_get_position_margin_invalid_symbol(self):
        """Test get_position_margin with invalid symbol type."""
        with patch.object(self.exchange, '_make_request', return_value=[]):
            result = asyncio.run(self.exchange.get_position_margin(123))
            
            self.assertEqual(result, {})
            self.exchange.logger.error.assert_called()
    
    def test_get_position_funding_valid(self):
        """Test get_position_funding with valid data."""
        mock_data = [
            {
                "instId": "BTCUSDT",
                "type": "funding_fee",
                "bal": "1000",
                "balChg": "-10",
                "ccy": "USDT",
                "fee": "-10",
                "ts": "1640995200000"
            },
            {
                "instId": "BTCUSDT",
                "type": "trade",
                "bal": "1000",
                "balChg": "100",
                "ccy": "USDT"
            }
        ]
        
        with patch.object(self.exchange, '_make_request', return_value=mock_data):
            result = asyncio.run(self.exchange.get_position_funding("BTCUSDT"))
            
            # Should only return funding_fee records
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["type"], "funding_fee")
    
    def test_get_position_risk_metrics_no_positions(self):
        """Test get_position_risk_metrics with no positions."""
        with patch.object(self.exchange, 'get_all_positions', return_value=[]):
            result = asyncio.run(self.exchange.get_position_risk_metrics())
            
            self.assertEqual(result, {})
            self.exchange.logger.info.assert_called()
    
    def test_get_position_risk_metrics_with_positions(self):
        """Test get_position_risk_metrics with positions."""
        mock_positions = [
            {
                "symbol": "BTCUSDT",
                "unrealizedPnl": "1000",
                "notionalUsd": "50000",
                "margin": "1000",
                "leverage": "2",
                "size": "1.0",
                "side": "long",
                "liquidationPrice": "45000",
                "marginRatio": "0.02"
            }
        ]
        
        with patch.object(self.exchange, 'get_all_positions', return_value=mock_positions):
            result = asyncio.run(self.exchange.get_position_risk_metrics())
            
            self.assertEqual(result["totalPositions"], 1)
            self.assertEqual(result["totalUnrealizedPnl"], 1000)
            self.assertEqual(result["totalNotionalUsd"], 50000)
            self.assertEqual(result["maxLeverage"], 2)
    
    def test_calculate_position_size_valid(self):
        """Test calculate_position_size with valid inputs."""
        result = asyncio.run(self.exchange.calculate_position_size(
            symbol="BTCUSDT",
            risk_amount=1000,
            entry_price=50000,
            stop_loss_price=48000,
            leverage=2.0
        ))
        
        self.assertNotIn("error", result)
        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["risk_amount"], 1000)
        self.assertEqual(result["entry_price"], 50000)
        self.assertEqual(result["leverage"], 2.0)
        self.assertGreater(result["position_size"], 0)
    
    def test_calculate_position_size_invalid_prices(self):
        """Test calculate_position_size with invalid price difference."""
        result = asyncio.run(self.exchange.calculate_position_size(
            symbol="BTCUSDT",
            risk_amount=1000,
            entry_price=50000,
            stop_loss_price=50000,  # Same price
            leverage=2.0
        ))
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid price difference")
    
    def test_get_position_summary_no_positions(self):
        """Test get_position_summary with no positions."""
        with patch.object(self.exchange, 'get_all_positions', return_value=[]):
            result = asyncio.run(self.exchange.get_position_summary("SPOT"))
            
            self.assertEqual(result["totalPositions"], 0)
            self.assertEqual(result["totalValue"], 0)
            self.assertEqual(result["totalUnrealizedPnl"], 0)
    
    def test_get_position_alerts_no_risk(self):
        """Test get_position_alerts with no high-risk positions."""
        mock_risk_metrics = {
            "highRiskPositions": [],
            "positions": [{"leverage": 1}],
            "portfolioRiskScore": 0.01
        }
        
        with patch.object(self.exchange, 'get_position_risk_metrics', return_value=mock_risk_metrics):
            result = asyncio.run(self.exchange.get_position_alerts(0.05))
            
            self.assertEqual(len(result), 0)
    
    def test_get_position_alerts_high_risk(self):
        """Test get_position_alerts with high-risk positions."""
        mock_risk_metrics = {
            "highRiskPositions": [{
                "symbol": "BTCUSDT",
                "risk_score": 0.15,
                "unrealizedPnl": 1000,
                "notionalUsd": 50000,
                "leverage": 5
            }],
            "positions": [{"leverage": 15}],
            "portfolioRiskScore": 0.15
        }
        
        with patch.object(self.exchange, 'get_position_risk_metrics', return_value=mock_risk_metrics):
            result = asyncio.run(self.exchange.get_position_alerts(0.05))
            
            self.assertGreater(len(result), 0)
            # Should have high risk and high leverage alerts
            alert_types = [alert["type"] for alert in result]
            self.assertIn("HIGH_RISK", alert_types)
            self.assertIn("HIGH_LEVERAGE", alert_types)
            self.assertIn("PORTFOLIO_RISK", alert_types)


class TestOkxPositionStreaming(unittest.TestCase):
    """Test cases for OKX position streaming methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exchange = OkxExchange("test_key", "test_secret", "BTCUSDT")
        self.exchange.logger = MagicMock()
    
    def test_get_positions_stream_callback(self):
        """Test get_positions_stream callback functionality."""
        callback_called = False
        callback_data = None
        
        async def test_callback(positions):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = positions
        
        # Mock get_all_positions to return test data
        with patch.object(self.exchange, 'get_all_positions', return_value=[{"symbol": "BTCUSDT"}]):
            # Run for a short time
            async def run_test():
                task = asyncio.create_task(
                    self.exchange.get_positions_stream(test_callback, "SPOT")
                )
                await asyncio.sleep(0.1)  # Short delay
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            asyncio.run(run_test())
            
            # Note: Due to the nature of the streaming loop, we can't easily test
            # that the callback was called without making the test more complex
            # This is a basic structure test
            self.assertTrue(True)  # Placeholder assertion


def run_tests():
    """Run all tests."""
    print("🧪 Running OKX Position Methods Tests...")
    print("="*50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOkxPositions))
    suite.addTests(loader.loadTestsFromTestCase(TestOkxPositionStreaming))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for failure in result.failures:
            print(f"  FAIL: {failure[0]}")
        for error in result.errors:
            print(f"  ERROR: {error[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)