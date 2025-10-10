"""
Unit tests for RiskCalculator.
"""

import pytest
import math
from datetime import datetime
from unittest.mock import MagicMock

from exchanges.shared.risk.risk_calculator import (
    RiskCalculator, PositionRisk, PortfolioRisk, RiskLevel
)


class TestRiskCalculator:
    """Test cases for RiskCalculator."""

    @pytest.fixture
    def risk_calculator(self):
        """Create RiskCalculator instance for testing."""
        return RiskCalculator("test_exchange")

    def test_initialization(self, risk_calculator):
        """Test RiskCalculator initialization."""
        assert risk_calculator.exchange_name == "test_exchange"
        assert risk_calculator.margin_ratio_warning == 0.8
        assert risk_calculator.margin_ratio_critical == 0.9
        assert risk_calculator.margin_ratio_liquidation == 0.95
        assert risk_calculator.default_initial_margin == 0.1
        assert risk_calculator.default_maintenance_margin == 0.05

    def test_calculate_position_risk_long(self, risk_calculator):
        """Test calculating risk for long position."""
        position_risk = risk_calculator.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0
        )
        
        assert position_risk.symbol == "BTCUSDT"
        assert position_risk.position_size == 0.1
        assert position_risk.entry_price == 50000.0
        assert position_risk.current_price == 51000.0
        assert position_risk.leverage == 2.0
        assert position_risk.unrealized_pnl == 100.0  # 0.1 * (51000 - 50000)
        assert position_risk.notional_value == 5100.0  # 0.1 * 51000
        assert position_risk.initial_margin == 0.1
        assert position_risk.maintenance_margin == 0.05

    def test_calculate_position_risk_short(self, risk_calculator):
        """Test calculating risk for short position."""
        position_risk = risk_calculator.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=-0.1,
            entry_price=50000.0,
            current_price=49000.0,
            leverage=2.0
        )
        
        assert position_risk.position_size == -0.1
        assert position_risk.unrealized_pnl == 100.0  # 0.1 * (50000 - 49000)
        assert position_risk.notional_value == 4900.0  # 0.1 * 49000

    def test_calculate_position_risk_custom_margins(self, risk_calculator):
        """Test calculating risk with custom margins."""
        position_risk = risk_calculator.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0,
            initial_margin=0.05,
            maintenance_margin=0.02
        )
        
        assert position_risk.initial_margin == 0.05
        assert position_risk.maintenance_margin == 0.02

    def test_calculate_liquidation_price_long(self, risk_calculator):
        """Test calculating liquidation price for long position."""
        liquidation_price = risk_calculator._calculate_liquidation_price(
            entry_price=50000.0,
            position_size=0.1,
            leverage=2.0,
            initial_margin=0.1,
            maintenance_margin=0.05
        )
        
        # For long: liquidation_price = entry_price * (1 - initial_margin + maintenance_margin) / leverage
        expected = 50000.0 * (1 - 0.1 + 0.05) / 2.0
        assert liquidation_price == expected

    def test_calculate_liquidation_price_short(self, risk_calculator):
        """Test calculating liquidation price for short position."""
        liquidation_price = risk_calculator._calculate_liquidation_price(
            entry_price=50000.0,
            position_size=-0.1,
            leverage=2.0,
            initial_margin=0.1,
            maintenance_margin=0.05
        )
        
        # For short: liquidation_price = entry_price * (1 + initial_margin - maintenance_margin) / leverage
        expected = 50000.0 * (1 + 0.1 - 0.05) / 2.0
        assert liquidation_price == expected

    def test_calculate_liquidation_price_zero_position(self, risk_calculator):
        """Test calculating liquidation price for zero position."""
        liquidation_price = risk_calculator._calculate_liquidation_price(
            entry_price=50000.0,
            position_size=0.0,
            leverage=2.0,
            initial_margin=0.1,
            maintenance_margin=0.05
        )
        
        assert liquidation_price == 0.0

    def test_determine_risk_level_low(self, risk_calculator):
        """Test determining low risk level."""
        risk_level = risk_calculator._determine_risk_level(0.5)
        assert risk_level == RiskLevel.LOW

    def test_determine_risk_level_medium(self, risk_calculator):
        """Test determining medium risk level."""
        risk_level = risk_calculator._determine_risk_level(0.85)
        assert risk_level == RiskLevel.MEDIUM

    def test_determine_risk_level_high(self, risk_calculator):
        """Test determining high risk level."""
        risk_level = risk_calculator._determine_risk_level(0.92)
        assert risk_level == RiskLevel.HIGH

    def test_determine_risk_level_critical(self, risk_calculator):
        """Test determining critical risk level."""
        risk_level = risk_calculator._determine_risk_level(0.96)
        assert risk_level == RiskLevel.CRITICAL

    def test_calculate_portfolio_risk(self, risk_calculator):
        """Test calculating portfolio risk."""
        positions = [
            PositionRisk(
                symbol="BTCUSDT",
                position_size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                leverage=2.0,
                margin_used=255.0,
                unrealized_pnl=100.0,
                margin_ratio=0.5,
                liquidation_price=45000.0,
                risk_level=RiskLevel.LOW,
                maintenance_margin=0.05,
                initial_margin=0.1,
                notional_value=5100.0
            ),
            PositionRisk(
                symbol="ETHUSDT",
                position_size=1.0,
                entry_price=3000.0,
                current_price=3100.0,
                leverage=3.0,
                margin_used=103.33,
                unrealized_pnl=100.0,
                margin_ratio=0.3,
                liquidation_price=2800.0,
                risk_level=RiskLevel.LOW,
                maintenance_margin=0.05,
                initial_margin=0.1,
                notional_value=3100.0
            )
        ]
        
        portfolio_risk = risk_calculator.calculate_portfolio_risk(positions, 10000.0)
        
        assert portfolio_risk.total_equity == 10000.0
        assert portfolio_risk.total_margin_used == 358.33
        assert portfolio_risk.total_unrealized_pnl == 200.0
        assert portfolio_risk.max_leverage_used == 3.0
        assert portfolio_risk.total_notional == 8200.0

    def test_calculate_var(self, risk_calculator):
        """Test calculating Value at Risk."""
        positions = [
            PositionRisk(
                symbol="BTCUSDT",
                position_size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                leverage=2.0,
                margin_used=255.0,
                unrealized_pnl=100.0,
                margin_ratio=0.5,
                liquidation_price=45000.0,
                risk_level=RiskLevel.LOW,
                maintenance_margin=0.05,
                initial_margin=0.1,
                notional_value=5100.0
            )
        ]
        
        var = risk_calculator.calculate_var(positions, confidence_level=0.95, time_horizon=1)
        
        assert var > 0
        assert var < 5100.0  # Should be less than portfolio value

    def test_calculate_var_empty_positions(self, risk_calculator):
        """Test calculating VaR with empty positions."""
        var = risk_calculator.calculate_var([])
        assert var == 0.0

    def test_get_z_score(self, risk_calculator):
        """Test getting Z-score for confidence levels."""
        assert risk_calculator._get_z_score(0.90) == 1.28
        assert risk_calculator._get_z_score(0.95) == 1.65
        assert risk_calculator._get_z_score(0.99) == 2.33
        assert risk_calculator._get_z_score(0.999) == 3.09
        assert risk_calculator._get_z_score(0.50) == 1.65  # Default

    def test_calculate_max_position_size(self, risk_calculator):
        """Test calculating maximum position size."""
        max_size = risk_calculator.calculate_max_position_size(
            symbol="BTCUSDT",
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0,
            available_margin=1000.0,
            risk_tolerance=0.8
        )
        
        # max_notional = available_margin * leverage / initial_margin * risk_tolerance
        # max_position_size = max_notional / current_price
        expected = (1000.0 * 2.0 / 0.1 * 0.8) / 51000.0
        assert abs(max_size - expected) < 0.0001

    def test_calculate_margin_requirement(self, risk_calculator):
        """Test calculating margin requirement."""
        required_margin = risk_calculator.calculate_margin_requirement(
            symbol="BTCUSDT",
            position_size=0.1,
            current_price=50000.0,
            leverage=2.0
        )
        
        # required_margin = notional_value * initial_margin / leverage
        expected = (0.1 * 50000.0) * 0.1 / 2.0
        assert required_margin == expected

    def test_calculate_margin_requirement_custom_initial_margin(self, risk_calculator):
        """Test calculating margin requirement with custom initial margin."""
        required_margin = risk_calculator.calculate_margin_requirement(
            symbol="BTCUSDT",
            position_size=0.1,
            current_price=50000.0,
            leverage=2.0,
            initial_margin=0.05
        )
        
        expected = (0.1 * 50000.0) * 0.05 / 2.0
        assert required_margin == expected

    def test_calculate_liquidation_distance_long(self, risk_calculator):
        """Test calculating liquidation distance for long position."""
        position_risk = PositionRisk(
            symbol="BTCUSDT",
            position_size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0,
            margin_used=255.0,
            unrealized_pnl=100.0,
            margin_ratio=0.5,
            liquidation_price=45000.0,
            risk_level=RiskLevel.LOW,
            maintenance_margin=0.05,
            initial_margin=0.1,
            notional_value=5100.0
        )
        
        distance = risk_calculator.calculate_liquidation_distance(position_risk)
        
        # distance = (current_price - liquidation_price) / current_price * 100
        expected = (51000.0 - 45000.0) / 51000.0 * 100
        assert abs(distance - expected) < 0.01

    def test_calculate_liquidation_distance_short(self, risk_calculator):
        """Test calculating liquidation distance for short position."""
        position_risk = PositionRisk(
            symbol="BTCUSDT",
            position_size=-0.1,
            entry_price=50000.0,
            current_price=49000.0,
            leverage=2.0,
            margin_used=245.0,
            unrealized_pnl=100.0,
            margin_ratio=0.5,
            liquidation_price=55000.0,
            risk_level=RiskLevel.LOW,
            maintenance_margin=0.05,
            initial_margin=0.1,
            notional_value=4900.0
        )
        
        distance = risk_calculator.calculate_liquidation_distance(position_risk)
        
        # distance = (liquidation_price - current_price) / current_price * 100
        expected = (55000.0 - 49000.0) / 49000.0 * 100
        assert abs(distance - expected) < 0.01

    def test_calculate_liquidation_distance_zero_position(self, risk_calculator):
        """Test calculating liquidation distance for zero position."""
        position_risk = PositionRisk(
            symbol="BTCUSDT",
            position_size=0.0,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0,
            margin_used=0.0,
            unrealized_pnl=0.0,
            margin_ratio=0.0,
            liquidation_price=0.0,
            risk_level=RiskLevel.LOW,
            maintenance_margin=0.05,
            initial_margin=0.1,
            notional_value=0.0
        )
        
        distance = risk_calculator.calculate_liquidation_distance(position_risk)
        
        assert distance == 100.0

    def test_get_risk_summary(self, risk_calculator):
        """Test getting risk summary."""
        positions = [
            PositionRisk(
                symbol="BTCUSDT",
                position_size=0.1,
                entry_price=50000.0,
                current_price=51000.0,
                leverage=2.0,
                margin_used=255.0,
                unrealized_pnl=100.0,
                margin_ratio=0.5,
                liquidation_price=45000.0,
                risk_level=RiskLevel.LOW,
                maintenance_margin=0.05,
                initial_margin=0.1,
                notional_value=5100.0
            ),
            PositionRisk(
                symbol="ETHUSDT",
                position_size=1.0,
                entry_price=3000.0,
                current_price=3100.0,
                leverage=3.0,
                margin_used=103.33,
                unrealized_pnl=100.0,
                margin_ratio=0.95,  # High risk
                liquidation_price=2800.0,
                risk_level=RiskLevel.HIGH,
                maintenance_margin=0.05,
                initial_margin=0.1,
                notional_value=3100.0
            )
        ]
        
        portfolio_risk = PortfolioRisk(
            total_equity=10000.0,
            total_margin_used=358.33,
            total_unrealized_pnl=200.0,
            portfolio_margin_ratio=0.5,
            risk_level=RiskLevel.LOW,
            positions=positions,
            max_leverage_used=3.0,
            total_notional=8200.0
        )
        
        summary = risk_calculator.get_risk_summary(portfolio_risk)
        
        assert summary["total_equity"] == 10000.0
        assert summary["total_margin_used"] == 358.33
        assert summary["total_unrealized_pnl"] == 200.0
        assert summary["portfolio_margin_ratio"] == 0.5
        assert summary["risk_level"] == "low"
        assert summary["max_leverage_used"] == 3.0
        assert summary["total_notional"] == 8200.0
        assert summary["high_risk_positions"] == 1
        assert summary["total_positions"] == 2

    def test_set_risk_thresholds(self, risk_calculator):
        """Test setting risk thresholds."""
        risk_calculator.set_risk_thresholds(
            warning_ratio=0.7,
            critical_ratio=0.8,
            liquidation_ratio=0.9
        )
        
        assert risk_calculator.margin_ratio_warning == 0.7
        assert risk_calculator.margin_ratio_critical == 0.8
        assert risk_calculator.margin_ratio_liquidation == 0.9

    def test_set_default_margins(self, risk_calculator):
        """Test setting default margins."""
        risk_calculator.set_default_margins(
            initial_margin=0.05,
            maintenance_margin=0.02
        )
        
        assert risk_calculator.default_initial_margin == 0.05
        assert risk_calculator.default_maintenance_margin == 0.02

    def test_validate_position_risk_safe(self, risk_calculator):
        """Test validating safe position risk."""
        position_risk = PositionRisk(
            symbol="BTCUSDT",
            position_size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0,
            margin_used=255.0,
            unrealized_pnl=100.0,
            margin_ratio=0.5,
            liquidation_price=45000.0,
            risk_level=RiskLevel.LOW,
            maintenance_margin=0.05,
            initial_margin=0.1,
            notional_value=5100.0
        )
        
        is_safe, warnings = risk_calculator.validate_position_risk(position_risk)
        
        assert is_safe is True
        assert len(warnings) == 0

    def test_validate_position_risk_warning(self, risk_calculator):
        """Test validating position risk with warnings."""
        position_risk = PositionRisk(
            symbol="BTCUSDT",
            position_size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=15.0,  # High leverage
            margin_used=255.0,
            unrealized_pnl=100.0,
            margin_ratio=0.85,  # Warning level
            liquidation_price=45000.0,
            risk_level=RiskLevel.MEDIUM,
            maintenance_margin=0.05,
            initial_margin=0.1,
            notional_value=5100.0
        )
        
        is_safe, warnings = risk_calculator.validate_position_risk(position_risk)
        
        assert is_safe is True  # Not critical
        assert len(warnings) > 0
        assert any("WARNING" in warning for warning in warnings)

    def test_validate_position_risk_critical(self, risk_calculator):
        """Test validating position risk with critical warnings."""
        position_risk = PositionRisk(
            symbol="BTCUSDT",
            position_size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=2.0,
            margin_used=255.0,
            unrealized_pnl=100.0,
            margin_ratio=0.96,  # Critical level
            liquidation_price=45000.0,
            risk_level=RiskLevel.CRITICAL,
            maintenance_margin=0.05,
            initial_margin=0.1,
            notional_value=5100.0
        )
        
        is_safe, warnings = risk_calculator.validate_position_risk(position_risk)
        
        assert is_safe is False
        assert len(warnings) > 0
        assert any("CRITICAL" in warning for warning in warnings)