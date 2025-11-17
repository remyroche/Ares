"""
Comprehensive unit tests for SimulatorConfig module.

Tests configuration management, fee structure, spread calculation,
validation, and conversion functionality.
"""

import pytest
import math
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from src.simulator.config import SimulatorConfig, SlippageModel
from tests.conftest import TPRINT_AVAILABLE, tprint_logged, LogLevel

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure,
    assert_execution_time,
    assert_timestamp_format,
    assert_true, assert_equals, assert_not_equals, assert_greater_than,
    assert_less_than, assert_greater_than_or_equal, assert_less_than_or_equal,
    assert_is_instance, assert_is_not_none, assert_in, assert_not_in
)


class TestSlippageModel:
    """Test suite for SlippageModel enum."""
    
    def test_slippage_model_values(self):
        """Test SlippageModel enum values."""
        assert_equals(SlippageModel.ORDERBOOK.value, "orderbook", "La valeur ORDERBOOK doit être 'orderbook'", "Test SlippageModel enum values")
        assert_equals(SlippageModel.PERCENTAGE.value, "percentage", "La valeur PERCENTAGE doit être 'percentage'", "Test SlippageModel enum values")
    
    def test_slippage_model_membership(self):
        """Test SlippageModel membership testing."""
        assert_true("orderbook" in [model.value for model in SlippageModel], "orderbook doit être présent dans les valeurs du modèle", "Test SlippageModel membership testing")
        assert_true("percentage" in [model.value for model in SlippageModel], "percentage doit être présent dans les valeurs du modèle", "Test SlippageModel membership testing")
    
    def test_slippage_model_iteration(self):
        """Test SlippageModel enum iteration."""
        models = list(SlippageModel)
        assert_equals(len(models), 2, "Le nombre de modèles doit être 2", "Test SlippageModel enum iteration")
        assert_true(SlippageModel.ORDERBOOK in models, "ORDERBOOK doit être dans la liste des modèles", "Test SlippageModel enum iteration")
        assert_true(SlippageModel.PERCENTAGE in models, "PERCENTAGE doit être dans la liste des modèles", "Test SlippageModel enum iteration")


class TestSimulatorConfig:
    """Test suite for SimulatorConfig class."""
    
    @pytest.fixture
    def default_config(self) -> SimulatorConfig:
        """Create default SimulatorConfig for testing."""
        return SimulatorConfig()
    
    @pytest.fixture
    def custom_config(self) -> SimulatorConfig:
        """Create custom SimulatorConfig for testing."""
        return SimulatorConfig(
            fee_structure={
                "binance": {"maker": 0.0005, "taker": 0.0009},
                "okx": {"maker": 0.0007, "taker": 0.0011},
                "custom_exchange": {"maker": 0.001, "taker": 0.002}
            },
            default_maker_fee=0.0004,
            default_taker_fee=0.0006,
            use_maker_taker_distinction=False,
            base_spread_bps=3.0,
            max_slippage_pct=0.02,
            max_positions_per_symbol=5,
            max_position_size_usd=100000.0,
            max_total_exposure_usd=200000.0
        )
    
    def test_default_config_creation(self, default_config: SimulatorConfig):
        """Test default SimulatorConfig creation."""
        assert_is_instance(default_config, SimulatorConfig, "La configuration par défaut doit être une instance de SimulatorConfig", "Test default SimulatorConfig creation")
        assert_true(default_config.use_maker_taker_distinction is True, "La distinction maker/taker doit être activée par défaut", "Test default SimulatorConfig creation")
        assert_float_equals(default_config.default_maker_fee, 0.0006, message="Le maker fee par défaut doit être 0.0006")
        assert_float_equals(default_config.default_taker_fee, 0.0008, message="Le taker fee par défaut doit être 0.0008")
        assert_float_equals(default_config.base_spread_bps, 2.0, message="Le spread de base par défaut doit être 2.0 bps")
        assert_float_equals(default_config.max_slippage_pct, 0.01, message="Le slippage maximal par défaut doit être 0.01")
        assert_float_equals(default_config.max_positions_per_symbol, 3, message="Le nombre max de positions par symbole par défaut doit être 3")
        assert_float_equals(default_config.max_position_size_usd, 50000.0, message="La taille max de position par défaut doit être 50000.0 USD")
        assert_float_equals(default_config.max_total_exposure_usd, 100000.0, message="L'exposition totale max par défaut doit être 100000.0 USD")
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def test_custom_config_creation(self, custom_config: SimulatorConfig):
        """Test custom SimulatorConfig creation with tprint tracing."""
        assert_true(custom_config.use_maker_taker_distinction is False, "La distinction maker/taker doit être désactivée pour la config personnalisée", "Test custom SimulatorConfig creation with tprint tracing")
        assert_float_equals(custom_config.default_maker_fee, 0.0004, message="Le maker fee personnalisé doit être 0.0004")
        assert_float_equals(custom_config.default_taker_fee, 0.0006, message="Le taker fee personnalisé doit être 0.0006")
        assert_float_equals(custom_config.base_spread_bps, 3.0, message="Le spread de base personnalisé doit être 3.0 bps")
        assert_float_equals(custom_config.max_slippage_pct, 0.02, message="Le slippage maximal personnalisé doit être 0.02")
        assert_float_equals(custom_config.max_positions_per_symbol, 5, message="Le nombre max de positions personnalisé doit être 5")
        assert_float_equals(custom_config.max_position_size_usd, 100000.0, message="La taille max de position personnalisée doit être 100000.0 USD")
        assert_float_equals(custom_config.max_total_exposure_usd, 200000.0, message="L'exposition totale max personnalisée doit être 200000.0 USD")
        return custom_config
    
    def test_fee_structure_defaults(self, default_config: SimulatorConfig):
        """Test default fee structure."""
        expected_exchanges = ["binance", "okx", "gateio", "mexc", "phemex"]
        assert_equals(set(default_config.fee_structure.keys()), set(expected_exchanges), "Les exchanges par défaut doivent correspondre à la liste attendue", "Test default fee structure")
        
        # Test binance fees
        binance_fees = default_config.fee_structure["binance"]
        assert_equals(binance_fees["maker"], 0.0006, "Le maker fee de Binance doit être 0.0006", "Test default fee structure")
        assert_equals(binance_fees["taker"], 0.001, "Le taker fee de Binance doit être 0.001", "Test default fee structure")
    
    def test_get_fee_rates_known_exchange(self, default_config: SimulatorConfig):
        """Test get_fee_rates for known exchanges."""
        # Test binance
        maker, taker = default_config.get_fee_rates("binance")
        assert_float_equals(maker, 0.0006, tolerance=1e-9, message="Le maker fee de Binance doit être 0.0006")
        assert_float_equals(taker, 0.001, tolerance=1e-9, message="Le taker fee de Binance doit être 0.001")
        
        # Test okx
        maker, taker = default_config.get_fee_rates("okx")
        assert_float_equals(maker, 0.0008, tolerance=1e-9, message="Le maker fee d'OKX doit être 0.0008")
        assert_float_equals(taker, 0.001, tolerance=1e-9, message="Le taker fee d'OKX doit être 0.001")
        
        # Test case insensitive
        maker, taker = default_config.get_fee_rates("BINANCE")
        assert_float_equals(maker, 0.0006, tolerance=1e-9, message="Le maker fee doit être insensible à la casse")
        assert_float_equals(taker, 0.001, tolerance=1e-9, message="Le taker fee doit être insensible à la casse")
    
    def test_get_fee_rates_unknown_exchange(self, default_config: SimulatorConfig):
        """Test get_fee_rates for unknown exchanges (should use defaults)."""
        maker, taker = default_config.get_fee_rates("unknown_exchange")
        assert_equals(maker, default_config.default_maker_fee, "Le maker fee pour exchange inconnu doit être celui par défaut", "Test get_fee_rates for unknown exchanges")
        assert_equals(taker, default_config.default_taker_fee, "Le taker fee pour exchange inconnu doit être celui par défaut", "Test get_fee_rates for unknown exchanges")
    
    def test_get_fee_rates_custom_defaults(self, custom_config: SimulatorConfig):
        """Test get_fee_rates with custom default fees."""
        maker, taker = custom_config.get_fee_rates("unknown_exchange")
        assert_float_equals(maker, custom_config.default_maker_fee, message="Le maker fee pour exchange inconnu doit être celui personnalisé par défaut")
        assert_float_equals(taker, custom_config.default_taker_fee, message="Le taker fee pour exchange inconnu doit être celui personnalisé par défaut")
    
    @pytest.mark.parametrize("exchange,expected_maker,expected_taker", [
        ("binance", 0.0006, 0.001),
        ("okx", 0.0008, 0.001),
        ("gateio", 0.0006, 0.001),
        ("mexc", 0.0007, 0.001),
        ("phemex", 0.0005, 0.001),
        ("unknown", 0.0006, 0.0008),  # Should use defaults
    ])
    def test_get_fee_rates_parametrized(
        self,
        default_config: SimulatorConfig,
        exchange: str,
        expected_maker: float,
        expected_taker: float
    ):
        """Test get_fee_rates with various exchanges."""
        maker, taker = default_config.get_fee_rates(exchange)
        assert_float_equals(maker, expected_maker, tolerance=1e-9, message=f"Le maker fee pour {exchange} doit être {expected_maker}")
        assert_float_equals(taker, expected_taker, tolerance=1e-9, message=f"Le taker fee pour {exchange} doit être {expected_taker}")
    
    def test_get_spread_pct_known_exchange(self, default_config: SimulatorConfig):
        """Test get_spread_pct for known exchanges."""
        # Test binance (multiplier 1.0)
        spread = default_config.get_spread_pct("binance")
        expected = (2.0 * 1.0) / 10000.0  # 2.0 bps * 1.0 / 10000
        assert_float_equals(spread, expected, tolerance=1e-9, message="Le spread pour Binance doit être calculé correctement", "Test get_spread_pct for known exchanges")
        
        # Test okx (multiplier 1.2)
        spread = default_config.get_spread_pct("okx")
        expected = (2.0 * 1.2) / 10000.0  # 2.0 bps * 1.2 / 10000
        assert_float_equals(spread, expected, tolerance=1e-9, message="Le spread pour OKX doit être calculé correctement avec le multiplicateur")
    
    def test_get_spread_pct_unknown_exchange(self, default_config: SimulatorConfig):
        """Test get_spread_pct for unknown exchanges (should use multiplier 1.0)."""
        spread = default_config.get_spread_pct("unknown_exchange")
        expected = (2.0 * 1.0) / 10000.0  # 2.0 bps * 1.0 / 10000
        assert_float_equals(spread, expected, tolerance=1e-9, message="Le spread pour exchange inconnu doit utiliser le multiplicateur par défaut")
    
    def test_get_spread_pct_custom_base_spread(self, custom_config: SimulatorConfig):
        """Test get_spread_pct with custom base spread."""
        spread = custom_config.get_spread_pct("binance")
        expected = (3.0 * 1.0) / 10000.0  # 3.0 bps * 1.0 / 10000
        assert_float_equals(spread, expected, tolerance=1e-9, message="Le spread personnalisé pour Binance doit être calculé correctement")
    
    @pytest.mark.parametrize("exchange,expected_multiplier,expected_spread", [
        ("binance", 1.0, 0.0002),  # 2.0 bps * 1.0 / 10000 = 0.0002
        ("okx", 1.2, 0.00024),    # 2.0 bps * 1.2 / 10000 = 0.00024
        ("gateio", 1.5, 0.0003),  # 2.0 bps * 1.5 / 10000 = 0.0003
        ("mexc", 1.8, 0.00036),   # 2.0 bps * 1.8 / 10000 = 0.00036
        ("unknown", 1.0, 0.0002), # 2.0 bps * 1.0 / 10000 = 0.0002
    ])
    def test_get_spread_pct_parametrized(
        self,
        default_config: SimulatorConfig,
        exchange: str,
        expected_multiplier: float,
        expected_spread: float
    ):
        """Test get_spread_pct with various exchanges."""
        spread = default_config.get_spread_pct(exchange)
        assert_float_equals(spread, expected_spread, tolerance=1e-9, message=f"Le spread pour {exchange} doit être {expected_spread}")
    
    def test_validate_success(self, default_config: SimulatorConfig):
        """Test successful validation."""
        result = default_config.validate()
        assert_true(result is True, "La validation par défaut doit réussir", "Test successful validation")
    
    def test_validate_negative_fees(self, default_config: SimulatorConfig):
        """Test validation with negative fees."""
        default_config.default_taker_fee = -0.001
        with pytest.raises(ValueError, match="Fee rates must be non-negative"):
            default_config.validate()
    
    def test_validate_invalid_slippage(self, default_config: SimulatorConfig):
        """Test validation with invalid slippage percentage."""
        default_config.max_slippage_pct = -0.01
        with pytest.raises(ValueError, match="max_slippage_pct must be between 0 and 1"):
            default_config.validate()
        
        default_config.max_slippage_pct = 1.5
        with pytest.raises(ValueError, match="max_slippage_pct must be between 0 and 1"):
            default_config.validate()
    
    def test_validate_invalid_positions(self, default_config: SimulatorConfig):
        """Test validation with invalid position limits."""
        default_config.max_positions_per_symbol = 0
        with pytest.raises(ValueError, match="max_positions_per_symbol must be at least 1"):
            default_config.validate()
    
    def test_validate_invalid_position_sizes(self, default_config: SimulatorConfig):
        """Test validation with invalid position size limits."""
        default_config.max_position_size_usd = 0
        with pytest.raises(ValueError, match="Position size limits must be positive"):
            default_config.validate()
        
        default_config.max_position_size_usd = 1000
        default_config.max_total_exposure_usd = 0
        with pytest.raises(ValueError, match="Position size limits must be positive"):
            default_config.validate()
    
    def test_validate_invalid_latency_range(self, default_config: SimulatorConfig):
        """Test validation with invalid latency range."""
        default_config.latency_range_ms = (-10, 100)
        with pytest.raises(ValueError, match="Invalid latency range"):
            default_config.validate()
        
        default_config.latency_range_ms = (100, 50)  # min > max
        with pytest.raises(ValueError, match="Invalid latency range"):
            default_config.validate()
    
    def test_to_dict_basic(self, default_config: SimulatorConfig):
        """Test basic to_dict conversion."""
        result = default_config.to_dict()
        assert_dict_structure(result, required_keys=[
            "fee_structure", "default_taker_fee", "default_maker_fee",
            "use_maker_taker_distinction", "slippage_model", "max_slippage_pct",
            "orderbook_depth_limit", "enable_latency_simulation", "latency_range_ms",
            "allow_multiple_positions", "allow_pyramiding", "max_positions_per_symbol",
            "allow_partial_closes", "max_position_size_usd", "max_total_exposure_usd",
            "orderbook_staleness_threshold_sec", "price_deviation_threshold_pct"
        ], message="Le dictionnaire doit contenir toutes les clés requises")
    
    def test_to_dict_values(self, default_config: SimulatorConfig):
        """Test to_dict value conversion."""
        result = default_config.to_dict()
        
        # Check some specific values
        assert_float_equals(result["default_taker_fee"], default_config.default_taker_fee, message="Le taker fee dans le dict doit correspondre")
        assert_float_equals(result["default_maker_fee"], default_config.default_maker_fee, message="Le maker fee dans le dict doit correspondre")
        assert result["use_maker_taker_distinction"] == default_config.use_maker_taker_distinction
        assert result["slippage_model"] == default_config.slippage_model.value
        assert_float_equals(result["max_slippage_pct"], default_config.max_slippage_pct, message="Le slippage max dans le dict doit correspondre")
        assert result["fee_structure"] == default_config.fee_structure
        assert result["latency_range_ms"] == default_config.latency_range_ms
    
    def test_to_dict_custom_config(self, custom_config: SimulatorConfig):
        """Test to_dict with custom configuration."""
        result = custom_config.to_dict()
        
        assert_float_equals(result["default_taker_fee"], custom_config.default_taker_fee, message="Le taker fee personnalisé dans le dict doit correspondre")
        assert_float_equals(result["default_maker_fee"], custom_config.default_maker_fee, message="Le maker fee personnalisé dans le dict doit correspondre")
        assert result["use_maker_taker_distinction"] == custom_config.use_maker_taker_distinction
        assert_float_equals(result["max_slippage_pct"], custom_config.max_slippage_pct, message="Le slippage max personnalisé dans le dict doit correspondre")
        assert result["fee_structure"] == custom_config.fee_structure
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def test_to_dict_with_tprint_tracing(self, default_config: SimulatorConfig):
        """Test to_dict conversion with tprint tracing."""
        result = default_config.to_dict()
        assert_is_instance(result, dict, "Le résultat doit être un dictionnaire", "Test to_dict conversion with tprint tracing")
        assert_true("fee_structure" in result, "Le dictionnaire doit contenir la clé fee_structure", "Test to_dict conversion with tprint tracing")
        return result
    
    def test_config_immutability_of_defaults(self):
        """Test that modifying default config doesn't affect new instances."""
        config1 = SimulatorConfig()
        original_maker_fee = config1.default_maker_fee
        
        # Modify config1
        config1.default_maker_fee = 0.001
        
        # Create new config - should have original defaults
        config2 = SimulatorConfig()
        assert_float_equals(config2.default_maker_fee, original_maker_fee, message="La nouvelle config doit avoir le maker fee par défaut")
        assert_float_equals(config2.default_maker_fee, original_maker_fee, message="La nouvelle config ne doit pas être affectée par la modification")
    
    def test_config_deep_copy_behavior(self):
        """Test that config objects behave correctly with deep copy scenarios."""
        config1 = SimulatorConfig()
        
        # Modify fee structure
        config1.fee_structure["binance"]["maker"] = 0.001
        
        # Create new config - should have original fee structure
        config2 = SimulatorConfig()
        assert_float_equals(config2.fee_structure["binance"]["maker"], 0.0006, message="La nouvelle config doit avoir le maker fee original")
        assert_float_equals(config2.fee_structure["binance"]["maker"], 0.0006, message="La nouvelle config ne doit pas être affectée par la modification")
    
    def test_config_spread_multiplier_modification(self, default_config: SimulatorConfig):
        """Test modifying spread multipliers."""
        # Modify binance spread multiplier
        default_config.spread_multiplier_by_exchange["binance"] = 2.0
        
        # Test that spread calculation uses new multiplier
        spread = default_config.get_spread_pct("binance")
        expected = (2.0 * 2.0) / 10000.0  # 2.0 bps * 2.0 / 10000
        assert_float_equals(spread, expected, tolerance=1e-9, message="Le spread modifié doit être calculé correctement")
    
    def test_config_fee_structure_modification(self, default_config: SimulatorConfig):
        """Test modifying fee structure."""
        # Add new exchange
        default_config.fee_structure["new_exchange"] = {"maker": 0.0015, "taker": 0.002}
        
        # Test that new exchange fees are used
        maker, taker = default_config.get_fee_rates("new_exchange")
        assert_equals(maker, 0.0015, "Le maker fee pour le nouvel exchange doit être 0.0015", "Test modifying fee structure")
        assert_equals(taker, 0.002, "Le taker fee pour le nouvel exchange doit être 0.002", "Test modifying fee structure")
    
    def test_config_validation_edge_cases(self, default_config: SimulatorConfig):
        """Test validation edge cases."""
        # Test boundary values
        default_config.max_slippage_pct = 0.0  # Minimum valid
        default_config.validate()  # Should not raise
        
        default_config.max_slippage_pct = 1.0  # Maximum valid
        default_config.validate()  # Should not raise
        
        default_config.max_positions_per_symbol = 1  # Minimum valid
        default_config.validate()  # Should not raise
        
        default_config.latency_range_ms = (0, 0)  # Minimum valid
        default_config.validate()  # Should not raise
    
    def test_config_dataclass_fields(self, default_config: SimulatorConfig):
        """Test that all expected dataclass fields are present."""
        import dataclasses
        
        fields = dataclasses.fields(default_config)
        field_names = [field.name for field in fields]
        
        expected_fields = [
            'fee_structure', 'default_taker_fee', 'default_maker_fee',
            'use_maker_taker_distinction', 'base_spread_bps',
            'spread_multiplier_by_exchange', 'slippage_model', 'max_slippage_pct',
            'orderbook_depth_limit', 'enable_latency_simulation',
            'latency_range_ms', 'allow_multiple_positions', 'allow_pyramiding',
            'max_positions_per_symbol', 'allow_partial_closes',
            'max_position_size_usd', 'max_total_exposure_usd',
            'orderbook_staleness_threshold_sec', 'price_deviation_threshold_pct'
        ]
        
        for field in expected_fields:
            assert_true(field in field_names, f"Le champ {field} doit être présent dans les champs du dataclass", "Test config dataclass fields")
    
    @pytest.mark.skipif(not TPRINT_AVAILABLE, reason="tprint not available")
    def test_tprint_integration(self, default_config: SimulatorConfig):
        """Test that tprint is properly integrated and working."""
        # This test ensures tprint calls don't fail
        maker, taker = default_config.get_fee_rates("binance")
        assert_equals(maker, 0.0006, "Le maker fee de Binance doit être 0.0006", "Test tprint integration")
        assert_equals(taker, 0.001, "Le taker fee de Binance doit être 0.001", "Test tprint integration")
        
        spread = default_config.get_spread_pct("binance")
        assert_greater_than(spread, 0, "Le spread doit être positif", "Test tprint integration")
        
        result = default_config.validate()
        assert_true(result is True, "La validation doit réussir", "Test tprint integration")
    
    def test_config_serialization_compatibility(self, default_config: SimulatorConfig):
        """Test that config can be serialized and deserialized."""
        import json
        
        # Convert to dict and serialize to JSON
        config_dict = default_config.to_dict()
        json_str = json.dumps(config_dict, default=str)
        
        # Deserialize from JSON
        loaded_dict = json.loads(json_str)
        
        # Verify key values are preserved
        assert_equals(loaded_dict["default_maker_fee"], default_config.default_maker_fee, "Le maker fee doit être préservé après sérialisation", "Test config serialization compatibility")
        assert_equals(loaded_dict["default_taker_fee"], default_config.default_taker_fee, "Le taker fee doit être préservé après sérialisation", "Test config serialization compatibility")
        assert_equals(loaded_dict["use_maker_taker_distinction"], default_config.use_maker_taker_distinction, "La distinction maker/taker doit être préservée après sérialisation", "Test config serialization compatibility")
        assert_equals(loaded_dict["max_slippage_pct"], default_config.max_slippage_pct, "Le slippage max doit être préservé après sérialisation", "Test config serialization compatibility")
        assert_equals(loaded_dict["fee_structure"], default_config.fee_structure, "La structure de frais doit être préservée après sérialisation", "Test config serialization compatibility")
    
    def test_config_with_custom_fee_structure(self):
        """Test creating config with completely custom fee structure."""
        custom_fees = {
            "exchange_a": {"maker": 0.001, "taker": 0.002},
            "exchange_b": {"maker": 0.0005, "taker": 0.0015},
            "exchange_c": {"maker": 0.002, "taker": 0.003}
        }
        
        config = SimulatorConfig(fee_structure=custom_fees)
        
        # Test that custom fees are used
        maker_a, taker_a = config.get_fee_rates("exchange_a")
        assert_float_equals(maker_a, 0.001, message="Le maker fee pour exchange_a doit être 0.001")
        assert_float_equals(taker_a, 0.002, message="Le taker fee pour exchange_a doit être 0.002")
        
        maker_b, taker_b = config.get_fee_rates("exchange_b")
        assert_float_equals(maker_b, 0.0005, message="Le maker fee pour exchange_b doit être 0.0005")
        assert_float_equals(taker_b, 0.0015, message="Le taker fee pour exchange_b doit être 0.0015")
        
        # Test that default exchanges are not present
        with pytest.raises(KeyError):
            _ = config.fee_structure["binance"]
