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
    assert_list_structure,
    assert_true, assert_equals, assert_not_equals, assert_greater_than,
    assert_less_than, assert_greater_than_or_equal, assert_less_than_or_equal,
    assert_is_instance, assert_is_not_none, assert_in, assert_not_in
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
        assert_equals(fee_calculator.config, config, "La configuration du calculateur doit correspondre à celle fournie", "Test basic initialization of FeeCalculator")
        assert_true(hasattr(fee_calculator, 'logger'), "Le calculateur doit avoir un attribut logger", "Test basic initialization of FeeCalculator")
        assert_is_instance(fee_calculator.logger, logging.Logger, "Le logger doit être une instance de logging.Logger", "Test basic initialization of FeeCalculator")
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def test_init_with_tprint_tracing(self, config: SimulatorConfig):
        """Test initialization with tprint tracing enabled."""
        calculator = FeeCalculator(config)
        assert_is_not_none(calculator, "Le calculateur ne doit pas être None", "Test initialization with tprint tracing enabled")
        assert_true(calculator.config.use_maker_taker_distinction is True, "La distinction maker/taker doit être activée", "Test initialization with tprint tracing enabled")
    
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
        
        assert_is_instance(result, FeeResult, "Le résultat doit être une instance de FeeResult", "Test basic fee calculation scenarios")
        assert_equals(result.fee_type, expected_fee_type, f"Le type de fee doit être {expected_fee_type}", "Test basic fee calculation scenarios")
        assert_float_equals(result.fee_percentage, expected_fee_rate, tolerance=1e-9, message=f"Le pourcentage de fee doit être {expected_fee_rate}")
        assert_equals(result.exchange, exchange, f"L'exchange doit être {exchange}", "Test basic fee calculation scenarios")
        assert_equals(result.is_maker, (expected_fee_type == "maker"), "Le statut is_maker doit correspondre au type de fee", "Test basic fee calculation scenarios")
        
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
        assert_equals(result.fee_type, "taker", "Le type de fee doit être 'taker' quand la distinction est désactivée", "Test fee calculation when maker/taker distinction is disabled")
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
        assert_equals(result.fee_type, "maker", "Le type de fee doit être 'maker' pour un exchange inconnu", "Test fee calculation for unknown exchange")
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
        assert_greater_than(result.fee_amount, 0, "Le montant de la fee doit être positif", "Test fee calculation precision")
    
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
        assert_in("binance", call_args, "Le log doit contenir le nom de l'exchange", "Test fee calculation logging")
        assert_in("limit", call_args, "Le log doit contenir le type d'ordre", "Test fee calculation logging")
        assert_in("maker fee=", call_args, "Le log doit contenir le maker fee", "Test fee calculation logging")
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def test_calculate_fee_with_tprint_tracing(self, fee_calculator: FeeCalculator):
        """Test fee calculation with tprint tracing enabled."""
        result = fee_calculator.calculate_fee(
            exchange="binance",
            quantity=1.0,
            price=100.0,
            order_type="limit"
        )
        assert_is_not_none(result, "Le résultat ne doit pas être None", "Test fee calculation with tprint tracing enabled")
        assert_is_instance(result, FeeResult, "Le résultat doit être une instance de FeeResult", "Test fee calculation with tprint tracing enabled")
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
        assert_equals(result["fee_type"], "maker/taker", "Le type de fee doit être 'maker/taker'", "Test total fee calculation for round trip trades")
    
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
        assert_equals(result["fee_type"], "maker/maker", "Le type de fee doit être 'maker/maker' pour deux makers", "Test total fee calculation when both fees are same type")
    
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
        assert_equals(result["fee_type"], "maker/taker", "Le type de fee doit être 'maker/taker' pour différents exchanges", "Test total fee calculation with different exchanges")
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
        assert_is_instance(result, dict, "Le résultat doit être un dictionnaire", "Test total fee calculation with tprint tracing enabled")
        assert_in("total_fee", result, "Le résultat doit contenir la clé 'total_fee'", "Test total fee calculation with tprint tracing enabled")
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
        assert_true(hasattr(result, 'fee_amount'), "Le résultat doit avoir l'attribut fee_amount", "Test FeeResult dataclass attributes")
        assert_true(hasattr(result, 'fee_percentage'), "Le résultat doit avoir l'attribut fee_percentage", "Test FeeResult dataclass attributes")
        assert_true(hasattr(result, 'fee_type'), "Le résultat doit avoir l'attribut fee_type", "Test FeeResult dataclass attributes")
        assert_true(hasattr(result, 'exchange'), "Le résultat doit avoir l'attribut exchange", "Test FeeResult dataclass attributes")
        assert_true(hasattr(result, 'is_maker'), "Le résultat doit avoir l'attribut is_maker", "Test FeeResult dataclass attributes")
        
        # Test attribute types
        assert_is_instance(result.fee_amount, float, "Le fee_amount doit être un float", "Test FeeResult dataclass attributes")
        assert_is_instance(result.fee_percentage, float, "Le fee_percentage doit être un float", "Test FeeResult dataclass attributes")
        assert_is_instance(result.fee_type, str, "Le fee_type doit être une chaîne", "Test FeeResult dataclass attributes")
        assert_is_instance(result.exchange, str, "L'exchange doit être une chaîne", "Test FeeResult dataclass attributes")
        assert_is_instance(result.is_maker, bool, "Le is_maker doit être un booléen", "Test FeeResult dataclass attributes")
        
        # Test attribute values
        assert_greater_than(result.fee_amount, 0, "Le fee_amount doit être positif", "Test FeeResult dataclass attributes")
        assert_greater_than(result.fee_percentage, 0, "Le fee_percentage doit être positif", "Test FeeResult dataclass attributes")
        assert_in(result.fee_type, ["maker", "taker"], "Le fee_type doit être 'maker' ou 'taker'", "Test FeeResult dataclass attributes")
        assert_greater_than(len(result.exchange), 0, "L'exchange ne doit pas être vide", "Test FeeResult dataclass attributes")
        assert_equals(result.is_maker, (result.fee_type == "maker"), "Le is_maker doit correspondre au fee_type", "Test FeeResult dataclass attributes")
    
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
        assert_equals(result.fee_type, "maker", "Le type de fee doit être 'maker' pour une quantité nulle", "Test edge case zero quantity")
    
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
        assert_equals(result.fee_type, "maker", "Le type de fee doit être 'maker' pour un prix nul", "Test edge case zero price")
    
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
        assert_greater_than(result.fee_amount, 0, "Le fee amount doit être positif pour les grandes valeurs", "Test edge case large values")
    
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
        assert_is_not_none(result, "Le résultat ne doit pas être None", "Test tprint integration")
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
        assert_greater_than(result2.fee_amount, result1.fee_amount, "Le nouveau fee amount doit être plus élevé", "Test config modification affects calculations")
        assert_greater_than(result2.fee_percentage, result1.fee_percentage, "Le nouveau fee percentage doit être plus élevé", "Test config modification affects calculations")


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
        assert_equals(result.fee_type, "maker", "Le type de fee doit être 'maker'", "Test FeeResult dataclass creation")
        assert_equals(result.exchange, "binance", "L'exchange doit être 'binance'", "Test FeeResult dataclass creation")
        assert_equals(result.is_maker, True, "Le is_maker doit être True", "Test FeeResult dataclass creation")
    
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
        assert_equals(result.fee_amount, 0.2, "Le fee_amount doit pouvoir être modifié", "Test FeeResult immutability")
    
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
        assert_in("FeeResult", str_repr, "La représentation doit contenir 'FeeResult'", "Test FeeResult string representation")
        assert_in("fee_amount=0.1", str_repr, "La représentation doit contenir le fee_amount", "Test FeeResult string representation")
        assert_in("fee_type='maker'", str_repr, "La représentation doit contenir le fee_type", "Test FeeResult string representation")
        assert_in("exchange='binance'", str_repr, "La représentation doit contenir l'exchange", "Test FeeResult string representation")
