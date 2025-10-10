"""
Test Exchange Integration

Comprehensive tests for the exchange integration system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.trading.integration.exchange_integration import (
    ExchangeIntegrationManager,
    ExchangeIntegrationConfig,
    create_exchange_integration,
    create_binance_integration,
    create_bingx_integration
)
from src.trading.execution.exchange_interface import TickerData, KlineData


class TestExchangeIntegrationConfig:
    """Test ExchangeIntegrationConfig."""
    
    def test_config_creation(self):
        """Test config creation with default values."""
        config = ExchangeIntegrationConfig(
            exchange_type="binance",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        assert config.exchange_type == "binance"
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.testnet is True
        assert config.trade_symbol == "BTCUSDT"
        assert config.password is None
        assert config.enable_shared_utilities is True
        assert config.enable_risk_management is True
        assert config.enable_rate_limiting is True
        assert config.rate_limits == {}
    
    def test_config_creation_custom(self):
        """Test config creation with custom values."""
        config = ExchangeIntegrationConfig(
            exchange_type="bingx",
            api_key="custom_key",
            api_secret="custom_secret",
            testnet=False,
            trade_symbol="ETHUSDT",
            password="test_password",
            enable_shared_utilities=False,
            enable_risk_management=False,
            enable_rate_limiting=False,
            rate_limits={"ticker": 50}
        )
        
        assert config.exchange_type == "bingx"
        assert config.api_key == "custom_key"
        assert config.api_secret == "custom_secret"
        assert config.testnet is False
        assert config.trade_symbol == "ETHUSDT"
        assert config.password == "test_password"
        assert config.enable_shared_utilities is False
        assert config.enable_risk_management is False
        assert config.enable_rate_limiting is False
        assert config.rate_limits == {"ticker": 50}


class TestExchangeIntegrationManager:
    """Test ExchangeIntegrationManager."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return ExchangeIntegrationConfig(
            exchange_type="binance",
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create test manager."""
        with patch('src.trading.integration.exchange_integration.create_binance_exchange'):
            with patch('src.trading.integration.exchange_integration.ExchangeInterface'):
                with patch('src.trading.integration.exchange_integration.HighLevelAuthManager'):
                    with patch('src.trading.integration.exchange_integration.HighLevelMarketManager'):
                        with patch('src.trading.integration.exchange_integration.HighLevelOrderManager'):
                            with patch('src.trading.integration.exchange_integration.HighLevelRiskManager'):
                                with patch('src.trading.integration.exchange_integration.HighLevelBalanceManager'):
                                    with patch('src.trading.integration.exchange_integration.HighLevelRateLimitManager'):
                                        return ExchangeIntegrationManager(config)
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.is_initialized is True
        assert manager.is_connected is False
        assert manager.last_error is None
    
    def test_unsupported_exchange_type(self):
        """Test unsupported exchange type."""
        config = ExchangeIntegrationConfig(
            exchange_type="unsupported",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        with pytest.raises(ValueError, match="Unsupported exchange type"):
            ExchangeIntegrationManager(config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, manager):
        """Test successful connection."""
        manager.exchange_interface.connect = AsyncMock(return_value=True)
        
        result = await manager.connect()
        
        assert result is True
        assert manager.is_connected is True
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, manager):
        """Test connection failure."""
        manager.exchange_interface.connect = AsyncMock(return_value=False)
        
        result = await manager.connect()
        
        assert result is False
        assert manager.is_connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, manager):
        """Test disconnection."""
        manager.is_connected = True
        manager.exchange_interface.disconnect = AsyncMock()
        
        await manager.disconnect()
        
        assert manager.is_connected is False
        manager.exchange_interface.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ticker_success(self, manager):
        """Test successful ticker retrieval."""
        manager.is_connected = True
        expected_ticker = TickerData(
            symbol="BTCUSDT",
            price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            bid_quantity=1.0,
            ask_quantity=1.0,
            volume_24h=1000000.0,
            price_change_24h=1000.0,
            price_change_percent_24h=2.0,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=datetime.now()
        )
        manager.exchange_interface.get_ticker = AsyncMock(return_value=expected_ticker)
        
        result = await manager.get_ticker("BTCUSDT")
        
        assert result == expected_ticker
        manager.exchange_interface.get_ticker.assert_called_once_with("BTCUSDT")
    
    @pytest.mark.asyncio
    async def test_get_ticker_not_connected(self, manager):
        """Test ticker retrieval when not connected."""
        manager.is_connected = False
        
        result = await manager.get_ticker("BTCUSDT")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_klines_success(self, manager):
        """Test successful klines retrieval."""
        manager.is_connected = True
        expected_klines = [
            KlineData(
                symbol="BTCUSDT",
                interval="1h",
                timestamp=datetime.now(),
                open_price=50000.0,
                high_price=50100.0,
                low_price=49900.0,
                close_price=50050.0,
                volume=100.0,
                close_time=datetime.now(),
                quote_asset_volume=5005000.0,
                number_of_trades=50,
                taker_buy_base_asset_volume=50.0,
                taker_buy_quote_asset_volume=2502500.0
            )
        ]
        manager.exchange_interface.get_klines = AsyncMock(return_value=expected_klines)
        
        result = await manager.get_klines("BTCUSDT", "1h", limit=1)
        
        assert result == expected_klines
        manager.exchange_interface.get_klines.assert_called_once_with(
            "BTCUSDT", "1h", None, None, 1
        )
    
    @pytest.mark.asyncio
    async def test_create_order_success(self, manager):
        """Test successful order creation."""
        manager.is_connected = True
        manager.risk_manager = Mock()
        manager.risk_manager.calculate_position_risk.return_value = {"risk_score": 0.5}
        manager.risk_manager.validate_risk_limits.return_value.is_valid = True
        
        expected_result = {
            "orderId": "test_order_123",
            "symbol": "BTCUSDT",
            "side": "buy",
            "type": "market",
            "quantity": 0.001,
            "status": "NEW"
        }
        manager.exchange_interface.create_order = AsyncMock(return_value=expected_result)
        
        result = await manager.create_order("BTCUSDT", "buy", "market", 0.001)
        
        assert result == expected_result
        manager.exchange_interface.create_order.assert_called_once_with(
            "BTCUSDT", "buy", "market", 0.001, None
        )
    
    @pytest.mark.asyncio
    async def test_create_order_risk_rejection(self, manager):
        """Test order creation with risk rejection."""
        manager.is_connected = True
        manager.risk_manager = Mock()
        manager.risk_manager.calculate_position_risk.return_value = {"risk_score": 0.9}
        manager.risk_manager.validate_risk_limits.return_value.is_valid = False
        
        result = await manager.create_order("BTCUSDT", "buy", "market", 0.001, price=50000.0)
        
        assert "error" in result
        assert result["error"] == "risk_limit_exceeded"
    
    @pytest.mark.asyncio
    async def test_create_order_not_connected(self, manager):
        """Test order creation when not connected."""
        manager.is_connected = False
        
        result = await manager.create_order("BTCUSDT", "buy", "market", 0.001)
        
        assert "error" in result
        assert result["error"] == "not_connected"
    
    @pytest.mark.asyncio
    async def test_get_account_balance_success(self, manager):
        """Test successful account balance retrieval."""
        manager.is_connected = True
        expected_balance = {"USDT": 1000.0, "BTC": 0.1}
        manager.exchange_interface.get_account_balance = AsyncMock(return_value=expected_balance)
        
        result = await manager.get_account_balance()
        
        assert result == expected_balance
        manager.exchange_interface.get_account_balance.assert_called_once_with(None)
    
    @pytest.mark.asyncio
    async def test_get_risk_info_success(self, manager):
        """Test successful risk info retrieval."""
        manager.is_connected = True
        expected_risk = {"risk_score": 0.5, "max_position": 0.1}
        manager.exchange_interface.get_risk_info = AsyncMock(return_value=expected_risk)
        
        result = await manager.get_risk_info("BTCUSDT", 0.1, 50000.0, 2.0)
        
        assert result == expected_risk
        manager.exchange_interface.get_risk_info.assert_called_once_with(
            "BTCUSDT", 0.1, 50000.0, 2.0
        )
    
    @pytest.mark.asyncio
    async def test_get_portfolio_risk_success(self, manager):
        """Test successful portfolio risk retrieval."""
        manager.is_connected = True
        positions = [{"symbol": "BTCUSDT", "size": 0.1, "price": 50000.0}]
        expected_risk = {"portfolio_risk": 0.3, "correlation": 0.8}
        manager.exchange_interface.get_portfolio_risk = AsyncMock(return_value=expected_risk)
        
        result = await manager.get_portfolio_risk(positions)
        
        assert result == expected_risk
        manager.exchange_interface.get_portfolio_risk.assert_called_once_with(positions)
    
    def test_get_status(self, manager):
        """Test status retrieval."""
        manager.is_initialized = True
        manager.is_connected = True
        manager.last_error = None
        
        status = manager.get_status()
        
        assert status["is_initialized"] is True
        assert status["is_connected"] is True
        assert status["exchange_type"] == "binance"
        assert status["last_error"] is None
        assert status["shared_utilities_enabled"] is True
        assert status["risk_management_enabled"] is True
        assert status["rate_limiting_enabled"] is True
    
    def test_reset(self, manager):
        """Test reset functionality."""
        manager.is_connected = True
        manager.last_error = "test_error"
        
        with patch.object(manager, '_initialize_integration'):
            manager.reset()
        
        assert manager.is_initialized is False
        assert manager.is_connected is False
        assert manager.last_error is None


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_binance_integration(self):
        """Test Binance integration creation."""
        with patch('src.trading.integration.exchange_integration.ExchangeIntegrationManager'):
            integration = create_binance_integration(
                api_key="test_key",
                api_secret="test_secret",
                testnet=True,
                trade_symbol="BTCUSDT"
            )
            assert integration is not None
    
    def test_create_bingx_integration(self):
        """Test BingX integration creation."""
        with patch('src.trading.integration.exchange_integration.ExchangeIntegrationManager'):
            integration = create_bingx_integration(
                api_key="test_key",
                api_secret="test_secret",
                testnet=True,
                trade_symbol="BTCUSDT"
            )
            assert integration is not None
    
    def test_create_exchange_integration(self):
        """Test custom exchange integration creation."""
        config = ExchangeIntegrationConfig(
            exchange_type="binance",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        with patch('src.trading.integration.exchange_integration.ExchangeIntegrationManager'):
            integration = create_exchange_integration(config)
            assert integration is not None


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        config = ExchangeIntegrationConfig(
            exchange_type="binance",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        with patch('src.trading.integration.exchange_integration.create_binance_exchange') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                ExchangeIntegrationManager(config)
    
    @pytest.mark.asyncio
    async def test_operation_error_handling(self):
        """Test operation error handling."""
        config = ExchangeIntegrationConfig(
            exchange_type="binance",
            api_key="test_key",
            api_secret="test_secret"
        )
        
        with patch('src.trading.integration.exchange_integration.create_binance_exchange'):
            with patch('src.trading.integration.exchange_integration.ExchangeInterface'):
                with patch('src.trading.integration.exchange_integration.HighLevelAuthManager'):
                    with patch('src.trading.integration.exchange_integration.HighLevelMarketManager'):
                        with patch('src.trading.integration.exchange_integration.HighLevelOrderManager'):
                            with patch('src.trading.integration.exchange_integration.HighLevelRiskManager'):
                                with patch('src.trading.integration.exchange_integration.HighLevelBalanceManager'):
                                    with patch('src.trading.integration.exchange_integration.HighLevelRateLimitManager'):
                                        manager = ExchangeIntegrationManager(config)
                                        
                                        # Test error handling in get_ticker
                                        manager.exchange_interface.get_ticker = AsyncMock(
                                            side_effect=Exception("API error")
                                        )
                                        
                                        result = await manager.get_ticker("BTCUSDT")
                                        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])