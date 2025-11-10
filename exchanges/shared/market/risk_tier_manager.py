"""
Risk Tier Management

Handles risk tiers, leverage limits, and position size restrictions per symbol.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint


class RiskTier(Enum):
    """Risk tier enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class RiskTierSpec:
    """Risk tier specification"""
    tier: RiskTier
    max_leverage: float
    max_position_size: float
    max_notional: float
    margin_ratio: float
    liquidation_ratio: float
    maintenance_margin: float
    initial_margin: float
    adl_tier: Optional[int] = None  # Auto-deleveraging tier
    risk_score: float = 0.0
    description: str = ""


@dataclass
class SymbolRiskProfile:
    """Symbol-specific risk profile"""
    symbol: str
    risk_tier: RiskTier
    max_leverage: float
    max_position_size: float
    max_notional: float
    margin_ratio: float
    liquidation_ratio: float
    maintenance_margin: float
    initial_margin: float
    adl_tier: Optional[int] = None
    risk_score: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class RiskTierManager:
    """
    Manages risk tiers and position size restrictions for symbols.
    """
    
    def __init__(self, exchange_name: str):
        tprint(f"Initializing RiskTierManager for exchange={exchange_name}", "INFO")
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"RiskTierManager.{exchange_name}")

        # Risk tier definitions
        self.risk_tiers: Dict[RiskTier, RiskTierSpec] = {}
        self.symbol_risk_profiles: Dict[str, SymbolRiskProfile] = {}

        # Initialize default risk tiers
        self._initialize_default_risk_tiers()
        tprint(f"RiskTierManager initialized successfully for {exchange_name}", "SUCCESS")
    
    def _initialize_default_risk_tiers(self) -> None:
        """Initialize default risk tier specifications."""
        self.risk_tiers = {
            RiskTier.LOW: RiskTierSpec(
                tier=RiskTier.LOW,
                max_leverage=2.0,
                max_position_size=1000000.0,
                max_notional=2000000.0,
                margin_ratio=0.5,
                liquidation_ratio=0.8,
                maintenance_margin=0.1,
                initial_margin=0.5,
                adl_tier=1,
                risk_score=1.0,
                description="Low risk tier with conservative limits"
            ),
            RiskTier.MEDIUM: RiskTierSpec(
                tier=RiskTier.MEDIUM,
                max_leverage=5.0,
                max_position_size=500000.0,
                max_notional=2500000.0,
                margin_ratio=0.2,
                liquidation_ratio=0.85,
                maintenance_margin=0.15,
                initial_margin=0.2,
                adl_tier=2,
                risk_score=2.0,
                description="Medium risk tier with moderate limits"
            ),
            RiskTier.HIGH: RiskTierSpec(
                tier=RiskTier.HIGH,
                max_leverage=10.0,
                max_position_size=250000.0,
                max_notional=2500000.0,
                margin_ratio=0.1,
                liquidation_ratio=0.9,
                maintenance_margin=0.2,
                initial_margin=0.1,
                adl_tier=3,
                risk_score=3.0,
                description="High risk tier with aggressive limits"
            ),
            RiskTier.VERY_HIGH: RiskTierSpec(
                tier=RiskTier.VERY_HIGH,
                max_leverage=20.0,
                max_position_size=100000.0,
                max_notional=2000000.0,
                margin_ratio=0.05,
                liquidation_ratio=0.95,
                maintenance_margin=0.25,
                initial_margin=0.05,
                adl_tier=4,
                risk_score=4.0,
                description="Very high risk tier with extreme limits"
            ),
            RiskTier.EXTREME: RiskTierSpec(
                tier=RiskTier.EXTREME,
                max_leverage=50.0,
                max_position_size=50000.0,
                max_notional=2500000.0,
                margin_ratio=0.02,
                liquidation_ratio=0.98,
                maintenance_margin=0.3,
                initial_margin=0.02,
                adl_tier=5,
                risk_score=5.0,
                description="Extreme risk tier with maximum limits"
            )
        }
    
    def set_risk_tier(self, tier: RiskTier, spec: RiskTierSpec) -> None:
        """Set risk tier specification."""
        tprint(f"Setting risk tier {tier.value} with max_leverage={spec.max_leverage}", "INFO")
        self.risk_tiers[tier] = spec
        self.logger.info(f"Set risk tier {tier.value} specification")
        tprint(f"Successfully set risk tier {tier.value}", "SUCCESS")
    
    def get_risk_tier(self, tier: RiskTier) -> Optional[RiskTierSpec]:
        """Get risk tier specification."""
        return self.risk_tiers.get(tier)
    
    def set_symbol_risk_profile(self, profile: SymbolRiskProfile) -> None:
        """Set symbol risk profile."""
        tprint(f"Setting risk profile for symbol={profile.symbol}, tier={profile.risk_tier.value}", "INFO")
        self.symbol_risk_profiles[profile.symbol] = profile
        self.logger.debug(f"Set risk profile for {profile.symbol}")
        tprint(f"Successfully set risk profile for {profile.symbol}", "SUCCESS")
    
    def get_symbol_risk_profile(self, symbol: str) -> Optional[SymbolRiskProfile]:
        """Get symbol risk profile."""
        return self.symbol_risk_profiles.get(symbol)
    
    def get_max_leverage(self, symbol: str) -> float:
        """Get maximum leverage for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.max_leverage
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].max_leverage
    
    def get_max_position_size(self, symbol: str) -> float:
        """Get maximum position size for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.max_position_size
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].max_position_size
    
    def get_max_notional(self, symbol: str) -> float:
        """Get maximum notional for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.max_notional
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].max_notional
    
    def get_margin_ratio(self, symbol: str) -> float:
        """Get margin ratio for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.margin_ratio
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].margin_ratio
    
    def get_liquidation_ratio(self, symbol: str) -> float:
        """Get liquidation ratio for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.liquidation_ratio
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].liquidation_ratio
    
    def get_maintenance_margin(self, symbol: str) -> float:
        """Get maintenance margin for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.maintenance_margin
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].maintenance_margin
    
    def get_initial_margin(self, symbol: str) -> float:
        """Get initial margin for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.initial_margin
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].initial_margin
    
    def get_adl_tier(self, symbol: str) -> Optional[int]:
        """Get ADL tier for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.adl_tier
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].adl_tier
    
    def get_risk_score(self, symbol: str) -> float:
        """Get risk score for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.risk_score
        
        # Default to low risk tier
        return self.risk_tiers[RiskTier.LOW].risk_score
    
    def get_risk_tier_for_symbol(self, symbol: str) -> RiskTier:
        """Get risk tier for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if profile:
            return profile.risk_tier
        
        # Default to low risk tier
        return RiskTier.LOW
    
    def validate_position_size(self, symbol: str, position_size: float) -> tuple[bool, str]:
        """
        Validate position size against risk limits.

        Args:
            symbol: Trading symbol
            position_size: Position size to validate

        Returns:
            (is_valid, error_message)
        """
        tprint(f"Validating position size: symbol={symbol}, size={position_size}", "INFO")
        max_size = self.get_max_position_size(symbol)
        if position_size > max_size:
            tprint(f"Position size validation failed: {position_size} > {max_size}", "ERROR")
            return False, f"Position size {position_size} exceeds maximum {max_size}"

        if position_size < 0:
            tprint(f"Position size validation failed: negative value {position_size}", "ERROR")
            return False, "Position size cannot be negative"

        tprint(f"Position size validation successful for {symbol}", "SUCCESS")
        return True, ""
    
    def validate_notional(self, symbol: str, notional: float) -> tuple[bool, str]:
        """
        Validate notional value against risk limits.

        Args:
            symbol: Trading symbol
            notional: Notional value to validate

        Returns:
            (is_valid, error_message)
        """
        tprint(f"Validating notional value: symbol={symbol}, notional={notional}", "INFO")
        max_notional = self.get_max_notional(symbol)
        if notional > max_notional:
            tprint(f"Notional validation failed: {notional} > {max_notional}", "ERROR")
            return False, f"Notional value {notional} exceeds maximum {max_notional}"

        if notional < 0:
            tprint(f"Notional validation failed: negative value {notional}", "ERROR")
            return False, "Notional value cannot be negative"

        tprint(f"Notional validation successful for {symbol}", "SUCCESS")
        return True, ""
    
    def validate_leverage(self, symbol: str, leverage: float) -> tuple[bool, str]:
        """
        Validate leverage against risk limits.

        Args:
            symbol: Trading symbol
            leverage: Leverage to validate

        Returns:
            (is_valid, error_message)
        """
        tprint(f"Validating leverage: symbol={symbol}, leverage={leverage}", "INFO")
        max_leverage = self.get_max_leverage(symbol)
        if leverage > max_leverage:
            tprint(f"Leverage validation failed: {leverage} > {max_leverage}", "ERROR")
            return False, f"Leverage {leverage} exceeds maximum {max_leverage}"

        if leverage < 1.0:
            tprint(f"Leverage validation failed: {leverage} < 1.0", "ERROR")
            return False, "Leverage must be at least 1.0"

        tprint(f"Leverage validation successful for {symbol}", "SUCCESS")
        return True, ""
    
    def calculate_required_margin(self, symbol: str, notional: float, leverage: float) -> float:
        """
        Calculate required margin for position.
        
        Args:
            symbol: Trading symbol
            notional: Position notional value
            leverage: Leverage used
            
        Returns:
            Required margin amount
        """
        initial_margin = self.get_initial_margin(symbol)
        return notional * initial_margin / leverage
    
    def calculate_maintenance_margin(self, symbol: str, notional: float) -> float:
        """
        Calculate maintenance margin for position.
        
        Args:
            symbol: Trading symbol
            notional: Position notional value
            
        Returns:
            Maintenance margin amount
        """
        maintenance_margin = self.get_maintenance_margin(symbol)
        return notional * maintenance_margin
    
    def calculate_liquidation_price(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        leverage: float
    ) -> float:
        """
        Calculate liquidation price for position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Position side (long/short)
            leverage: Leverage used
            
        Returns:
            Liquidation price
        """
        liquidation_ratio = self.get_liquidation_ratio(symbol)
        
        if side.lower() == "long":
            # For long positions: liquidation_price = entry_price * (1 - liquidation_ratio + 1/leverage)
            liquidation_price = entry_price * (1 - liquidation_ratio + 1 / leverage)
        else:
            # For short positions: liquidation_price = entry_price * (1 + liquidation_ratio - 1/leverage)
            liquidation_price = entry_price * (1 + liquidation_ratio - 1 / leverage)
        
        return max(0, liquidation_price)
    
    def get_symbols_by_risk_tier(self, risk_tier: RiskTier) -> List[str]:
        """Get symbols by risk tier."""
        return [
            symbol for symbol, profile in self.symbol_risk_profiles.items()
            if profile.risk_tier == risk_tier
        ]
    
    def get_high_risk_symbols(self) -> List[str]:
        """Get high risk symbols."""
        return (
            self.get_symbols_by_risk_tier(RiskTier.HIGH) +
            self.get_symbols_by_risk_tier(RiskTier.VERY_HIGH) +
            self.get_symbols_by_risk_tier(RiskTier.EXTREME)
        )
    
    def get_low_risk_symbols(self) -> List[str]:
        """Get low risk symbols."""
        return (
            self.get_symbols_by_risk_tier(RiskTier.LOW) +
            self.get_symbols_by_risk_tier(RiskTier.MEDIUM)
        )
    
    def update_symbol_risk_tier(self, symbol: str, new_tier: RiskTier) -> bool:
        """Update symbol risk tier."""
        tprint(f"Updating risk tier for symbol={symbol} to {new_tier.value}", "INFO")
        profile = self.get_symbol_risk_profile(symbol)
        if not profile:
            tprint(f"Failed to update risk tier: profile not found for {symbol}", "ERROR")
            return False

        tier_spec = self.get_risk_tier(new_tier)
        if not tier_spec:
            tprint(f"Failed to update risk tier: tier spec not found for {new_tier.value}", "ERROR")
            return False

        # Update profile with new tier specifications
        profile.risk_tier = new_tier
        profile.max_leverage = tier_spec.max_leverage
        profile.max_position_size = tier_spec.max_position_size
        profile.max_notional = tier_spec.max_notional
        profile.margin_ratio = tier_spec.margin_ratio
        profile.liquidation_ratio = tier_spec.liquidation_ratio
        profile.maintenance_margin = tier_spec.maintenance_margin
        profile.initial_margin = tier_spec.initial_margin
        profile.adl_tier = tier_spec.adl_tier
        profile.risk_score = tier_spec.risk_score
        profile.last_updated = datetime.now()

        self.logger.info(f"Updated {symbol} to risk tier {new_tier.value}")
        tprint(f"Successfully updated {symbol} to risk tier {new_tier.value}", "SUCCESS")
        return True
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get risk tier statistics."""
        total_symbols = len(self.symbol_risk_profiles)
        
        tier_counts = {}
        for profile in self.symbol_risk_profiles.values():
            tier = profile.risk_tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        high_risk_count = len(self.get_high_risk_symbols())
        low_risk_count = len(self.get_low_risk_symbols())
        
        return {
            "total_symbols": total_symbols,
            "tier_distribution": tier_counts,
            "high_risk_symbols": high_risk_count,
            "low_risk_symbols": low_risk_count,
            "available_risk_tiers": len(self.risk_tiers)
        }
    
    def get_symbol_risk_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive risk summary for symbol."""
        profile = self.get_symbol_risk_profile(symbol)
        if not profile:
            return None
        
        return {
            "symbol": symbol,
            "risk_tier": profile.risk_tier.value,
            "max_leverage": profile.max_leverage,
            "max_position_size": profile.max_position_size,
            "max_notional": profile.max_notional,
            "margin_ratio": profile.margin_ratio,
            "liquidation_ratio": profile.liquidation_ratio,
            "maintenance_margin": profile.maintenance_margin,
            "initial_margin": profile.initial_margin,
            "adl_tier": profile.adl_tier,
            "risk_score": profile.risk_score,
            "last_updated": profile.last_updated.isoformat()
        }
    
    def cleanup_old_profiles(self, max_age_days: int = 30) -> int:
        """Clean up old risk profiles."""
        tprint(f"Starting cleanup of old risk profiles (max_age_days={max_age_days})", "INFO")
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        symbols_to_remove = []
        for symbol, profile in self.symbol_risk_profiles.items():
            if profile.last_updated < cutoff_time:
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            del self.symbol_risk_profiles[symbol]
            cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old risk profiles")
            tprint(f"Successfully cleaned up {cleaned_count} old risk profiles", "SUCCESS")
        else:
            tprint("No old risk profiles to clean up", "INFO")

        return cleaned_count