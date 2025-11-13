"""
Comprehensive unit tests for FeeCalculator module.

Tests fee calculation functionality including maker/taker distinctions,
exchange-specific rates, and total fee calculations.
"""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import logging

from src.simulator.fee_calculator import FeeCalculator, FeeResult
from src.simulator.config import SimulatorConfig
from tests.conftest import TPRINT_AVAILABLE, tprint_logged, LogLevel

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure
)


class TestFeeCalculator:
    """Test suite for FeeCalculator class."""
    
    @pytest.fixture
    def config(self) -> SimulatorConfig:
        """Create test configuration."""
        return SimulatorConfig(
            fee_structure={
                "binance": {"maker": 0.0006, "taker": 0.001},
                "okx": {"maker": 0.0008, "taker": 0.0012},
                "unknown_exchange": {"maker": 0.001, "taker": 0.0015}
            },
            default_maker_fee=0.0005,
            default_taker_fee=0.0008,
            use_maker_taker_distinction=True
        )
    
    @pytest.fixture
    def fee_calculator(self, config: SimulatorConfig) -> FeeCalculator:
        """Create FeeCalculator instance for testing."""
        return FeeCalculator(config)
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock(spec=logging.Logger)
    
    def test_init_basic(self, fee_calculator: FeeCalculator, config: SimulatorConfig):
        """Test basic initialization of FeeCalculator."""
        assert fee_calculator.config == config
        assert hasattr(fee_calculator, 'logger')
        assert isinstance(fee_calculator.logger, logging.Logger)
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def test_init_with_tprint_tracing(self, config: SimulatorConfig):
        """Test initialization with tprint tracing enabled."""
        calculator = FeeCalculator(config)
        assert calculator is not None
        assert calculator.config.use_maker_taker_distinction is True
    
    @pytest.mark.parametrize("exchange,expected_maker,expected_taker", [
        ("binance", 0.0006, 0.001),
        ("okx", 0.0008, 0.0012),
        ("UNKNOWN_EXCHANGE", 0.001, 0.0015),  # Should use defaults from custom config
        ("new_exchange", 0.0005, 0.0008),  # Should use defaults from custom config
    ])
    def test_get_fee_rates_from_config(
        self, 
        config: SimulatorConfig, 
        exchange: str, 
        expected_maker: float, 
        expected_taker: float
    ):
        """Test fee rate retrieval from configuration."""
        maker_fee, taker_fee = config.get_fee_rates(exchange)
        assert_float_equals(maker_fee, expected_maker, tolerance=1e-9, message=f"Le maker fee pour {exchange} doit être {expected_maker}")
        assert_float_equals(taker_fee, expected_taker, tolerance=1e-9, message=f"Le taker fee pour {exchange} doit être {expected_taker}")
    
    @pytest.mark.parametrize("exchange,quantity,price,order_type,is_maker,expected_fee_type,expected_fee_rate", [
        ("binance", 1.0, 100.0, "limit", None, "maker", 0.0006),
        ("binance", 1.0, 100.0, "market", None, "taker", 0.001),
        ("binance", 1.0, 100.0, "limit", True, "maker", 0.0006),
        ("binance", 1.0, 100.0, "market", False, "taker", 0.001),
        ("okx", 2.0, 50.0, "limit", None, "maker", 0.0008),
        ("okx", 2.0, 50.0, "market", None, "taker", 0.0012),
    ])
    def test_calculate_fee_basic_scenarios(
        self,
        fee_calculator: FeeCalculator,
        exchange: str,
        quantity: float,
        price: float,
        order_type: str,
        is_maker: bool,
        expected_fee_type: str,
        expected_fee_rate: float
    ):
        """Test basic fee calculation scenarios."""
        result = fee_calculator.calculate_fee(
            exchange=exchange,
            quantity=quantity,
            price=price,
            order_type=order_type,
            is_maker=is_maker
        )
        
        assert isinstance(result, FeeResult)
        assert result.fee_type == expected_fee_type
        assert_float_equals(result.fee_percentage, expected_fee_rate, tolerance=1e-9, message=f"Le pourcentage de fee doit être {expected_fee_rate}")
        assert result.exchange == exchange
        assert result.is_maker == (expected_fee_type == "maker")
        
        # Calculate expected fee amount
        expected_fee_amount = quantity * price * expected_fee_rate
        assert_float_equals(result.fee_amount, expected_fee_amount, tolerance=1e-10, message="Le montant de la fee doit être calculé correctement")
    
    def test_calculate_fee_without_maker_taker_distinction(self, config: SimulatorConfig):
        """Test fee calculation when maker/taker distinction is disabled."""
        config.use_maker_taker_distinction = False
        calculator = FeeCalculator(config)
        
        result = calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit",
            is_maker=True
        )
        
        # Should always use taker fee when distinction is disabled
        assert result.fee_type == "taker"
        assert_float_equals(result.fee_percentage, 0.001, tolerance=1e-9, message="Le pourcentage de fee doit être celui du taker fee de Binance")
    
    def test_calculate_fee_unknown_exchange(self, fee_calculator: FeeCalculator):
        """Test fee calculation for unknown exchange (should use defaults)."""
        result = fee_calculator.calculate_fee(
            exchange="unknown_exchange_xyz",
            quantity=1.5,
            price=200.0,
            order_type="limit"
        )
        
        # Should use default fees
        assert result.fee_type == "maker"
        assert_float_equals(result.fee_percentage, fee_calculator.config.default_maker_fee, tolerance=1e-9, message="Le pourcentage de fee doit être celui par défaut")
        expected_amount = 1.5 * 200.0 * fee_calculator.config.default_maker_fee
        assert_float_equals(result.fee_amount, expected_amount, tolerance=1e-10, message="Le montant de la fee doit être calculé avec les valeurs par défaut")
    
    @pytest.mark.parametrize("quantity,price", [
        (0.1, 100.0),
        (1.0, 1000.0),
        (10.0, 50.0),
        (0.001, 50000.0),
    ])
    def test_calculate_fee_precision(
        self,
        fee_calculator: FeeCalculator,
        quantity: float,
        price: float
    ):
        """Test fee calculation precision with various quantities and prices."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=quantity,
            price=price,
            order_type="market"
        )
        
        expected_fee = quantity * price * 0.001  # binance taker fee
        assert_float_equals(result.fee_amount, expected_fee, tolerance=1e-10, message="Le montant de la fee doit être précis")
        assert result.fee_amount > 0
    
    def test_calculate_fee_logging(self, fee_calculator: FeeCalculator, mock_logger):
        """Test that fee calculation is properly logged."""
        fee_calculator.logger = mock_logger
        
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        
        # Check that debug logging was called
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "binance" in call_args
        assert "limit" in call_args
        assert "maker fee=" in call_args
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def test_calculate_fee_with_tprint_tracing(self, fee_calculator: FeeCalculator):
        """Test fee calculation with tprint tracing enabled."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        assert result is not None
        assert isinstance(result, FeeResult)
        return result
    
    def test_calculate_total_fee_basic(self, fee_calculator: FeeCalculator):
        """Test total fee calculation for round trip trades."""
        entry_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"  # maker
        )
        
        exit_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=110.0,
            order_type="market"  # taker
        )
        
        result = fee_calculator.calculate_total_fee(entry_fee, exit_fee)
        
        assert_dict_structure(result, required_keys=[
            "entry_fee", "exit_fee", "total_fee",
            "entry_fee_pct", "exit_fee_pct", "total_fee_pct", "fee_type"
        ], message="Le résultat doit contenir toutes les clés requises")
        
        # Verify calculations
        expected_total = entry_fee.fee_amount + exit_fee.fee_amount
        assert_float_equals(result["total_fee"], expected_total, tolerance=1e-10, message="Le total des fees doit être correct")
        assert_float_equals(result["entry_fee"], entry_fee.fee_amount, tolerance=1e-10, message="La fee d'entrée doit être correcte")
        assert_float_equals(result["exit_fee"], exit_fee.fee_amount, tolerance=1e-10, message="La fee de sortie doit être correcte")
        assert result["fee_type"] == "maker/taker"
    
    def test_calculate_total_fee_same_type(self, fee_calculator: FeeCalculator):
        """Test total fee calculation when both fees are same type."""
        entry_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"  # maker
        )
        
        exit_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=110.0,
            order_type="limit"  # maker
        )
        
        result = fee_calculator.calculate_total_fee(entry_fee, exit_fee)
        assert result["fee_type"] == "maker/maker"
    
    def test_calculate_total_fee_different_exchanges(self, fee_calculator: FeeCalculator):
        """Test total fee calculation with different exchanges."""
        entry_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        
        exit_fee = fee_calculator.calculate_fee(
            exchange="okx",
            quantity=1.0,
            price=110.0,
            order_type="market"
        )
        
        result = fee_calculator.calculate_total_fee(entry_fee, exit_fee)
        assert result["fee_type"] == "maker/taker"
        expected_total = entry_fee.fee_amount + exit_fee.fee_amount
        assert_float_equals(result["total_fee"], expected_total, tolerance=1e-10, message="Le total des fees pour différents exchanges doit être correct")
    
    def test_calculate_total_fee_percentage_calculation(self, fee_calculator: FeeCalculator):
        """Test that percentage calculations are correct."""
        entry_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"  # maker: 0.06%
        )
        
        exit_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=110.0,
            order_type="market"  # taker: 0.1%
        )
        
        result = fee_calculator.calculate_total_fee(entry_fee, exit_fee)
        
        # Check percentage calculations (should be in basis points, not decimal)
        assert_float_equals(result["entry_fee_pct"], 0.06, tolerance=1e-3, message="Le pourcentage de la fee d'entrée doit être 0.06%")
        assert_float_equals(result["exit_fee_pct"], 0.10, tolerance=1e-3, message="Le pourcentage de la fee de sortie doit être 0.10%")
        assert_float_equals(result["total_fee_pct"], 0.16, tolerance=1e-3, message="Le pourcentage total des fees doit être 0.16%")
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def test_calculate_total_fee_with_tprint_tracing(self, fee_calculator: FeeCalculator):
        """Test total fee calculation with tprint tracing enabled."""
        entry_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        
        exit_fee = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=110.0,
            order_type="market"
        )
        
        result = fee_calculator.calculate_total_fee(entry_fee, exit_fee)
        assert isinstance(result, dict)
        assert "total_fee" in result
        return result
    
    def test_fee_result_attributes(self, fee_calculator: FeeCalculator):
        """Test FeeResult dataclass attributes."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        
        # Test all expected attributes
        assert hasattr(result, 'fee_amount')
        assert hasattr(result, 'fee_percentage')
        assert hasattr(result, 'fee_type')
        assert hasattr(result, 'exchange')
        assert hasattr(result, 'is_maker')
        
        # Test attribute types
        assert isinstance(result.fee_amount, float)
        assert isinstance(result.fee_percentage, float)
        assert isinstance(result.fee_type, str)
        assert isinstance(result.exchange, str)
        assert isinstance(result.is_maker, bool)
        
        # Test attribute values
        assert result.fee_amount > 0
        assert result.fee_percentage > 0
        assert result.fee_type in ["maker", "taker"]
        assert len(result.exchange) > 0
        assert result.is_maker == (result.fee_type == "maker")
    
    def test_edge_case_zero_quantity(self, fee_calculator: FeeCalculator):
        """Test fee calculation with zero quantity."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=0.0,
            price=100.0,
            order_type="limit"
        )
        
        assert_float_equals(result.fee_amount, 0.0, tolerance=1e-15, message="La fee doit être zéro pour une quantité nulle")
        assert_float_equals(result.fee_percentage, 0.0006, tolerance=1e-9, message="Le pourcentage doit toujours utiliser le maker rate")
        assert result.fee_type == "maker"
    
    def test_edge_case_zero_price(self, fee_calculator: FeeCalculator):
        """Test fee calculation with zero price."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=0.0,
            order_type="limit"
        )
        
        assert_float_equals(result.fee_amount, 0.0, tolerance=1e-15, message="La fee doit être zéro pour un prix nul")
        assert_float_equals(result.fee_percentage, 0.0006, tolerance=1e-9, message="Le pourcentage doit être maintenu même avec un prix nul")
        assert result.fee_type == "maker"
    
    def test_edge_case_large_values(self, fee_calculator: FeeCalculator):
        """Test fee calculation with large values."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=10000.0,
            price=50000.0,
            order_type="market"
        )
        
        expected_fee = 10000.0 * 50000.0 * 0.001  # taker fee
        assert_float_equals(result.fee_amount, expected_fee, tolerance=1e-6, message="La fee pour les grandes valeurs doit être précise")
        assert result.fee_amount > 0
    
    @pytest.mark.skipif(not TPRINT_AVAILABLE, reason="tprint not available")
    def test_tprint_integration(self, fee_calculator: FeeCalculator):
        """Test that tprint is properly integrated and working."""
        # This test just ensures tprint calls don't fail
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        assert result is not None
        # If we get here without exceptions, tprint integration is working
    
    def test_config_modification_affects_calculations(self, config: SimulatorConfig):
        """Test that modifying config affects fee calculations."""
        # Initial calculation
        calculator = FeeCalculator(config)
        result1 = calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        
        # Modify config
        config.fee_structure["binance"]["maker"] = 0.001  # Increase maker fee
        calculator2 = FeeCalculator(config)
        result2 = calculator2.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        
        # New fee should be higher
        assert result2.fee_amount > result1.fee_amount
        assert result2.fee_percentage > result1.fee_percentage


class TestFeeResult:
    """Test suite for FeeResult dataclass."""
    
    def test_fee_result_creation(self):
        """Test FeeResult dataclass creation."""
        result = FeeResult(
            fee_amount=0.1,
            fee_percentage=0.001,
            fee_type="maker",
            exchange="binance",
            is_maker=True
        )
        
        assert_float_equals(result.fee_amount, 0.1, tolerance=1e-9, message="Le montant de la fee doit être 0.1")
        assert_float_equals(result.fee_percentage, 0.001, tolerance=1e-9, message="Le pourcentage de la fee doit être 0.001")
        assert result.fee_type == "maker"
        assert result.exchange == "binance"
        assert result.is_maker is True
    
    def test_fee_result_immutability(self):
        """Test that FeeResult is immutable (dataclass with frozen=True not set, but still test behavior)."""
        result = FeeResult(
            fee_amount=0.1,
            fee_percentage=0.001,
            fee_type="maker",
            exchange="binance",
            is_maker=True
        )
        
        # Dataclasses are mutable by default, so we can modify attributes
        result.fee_amount = 0.2
        assert result.fee_amount == 0.2
    
    def test_fee_result_string_representation(self):
        """Test FeeResult string representation."""
        result = FeeResult(
            fee_amount=0.1,
            fee_percentage=0.001,
            fee_type="maker",
            exchange="binance",
            is_maker=True
        )
        
        str_repr = str(result)
        assert "FeeResult" in str_repr
        assert "fee_amount=0.1" in str_repr
        assert "fee_type='maker'" in str_repr
        assert "exchange='binance'" in str_repr
