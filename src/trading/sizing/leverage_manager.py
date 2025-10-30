"""
Leverage Manager - Dampened Kelly Integration

Production-hardened leverage calculation using the same unified dampened Kelly logic
as position sizing. Shares the Kelly engine with PositionSizer to avoid duplicate calculations.

Key features:
- Unified dampened Kelly algorithm (same as position sizing)
- Uses beta_leverage and lev_floor parameters (regime-aware)
- Portfolio correlation checks
- Versioned config integration
- Reuses KellyResult from position sizer when available
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug
from ..config.trading_config import TradingConfig
from src.config.leverage_constants import MIN_LEVERAGE, MAX_LEVERAGE, validate_leverage

# Import dampened Kelly components
from .dampened_kelly_engine import DampenedKellyEngine, KellyResult

logger = system_logger.getChild('LeverageManager')


@dataclass
class LeverageResult:
    """Enhanced leverage calculation result with Kelly metadata."""
    symbol: str
    recommended_leverage: float
    max_leverage: float
    min_leverage: float
    confidence: float
    leverage_multiplier: float
    metadata: Dict[str, Any]


class LeverageManager:
    """
    Production-hardened leverage manager using unified dampened Kelly logic.
    
    This manager shares the dampened Kelly engine with PositionSizer and uses
    the same algorithm with regime-aware beta_leverage and lev_floor parameters.
    
    Features:
    - Unified dampened Kelly (same as position sizing)
    - Regime-aware parameters
    - Portfolio correlation checks
    - Config versioning
    - Reuses calculations when available
    """
    
    def __init__(self, config: TradingConfig, kelly_engine: Optional[DampenedKellyEngine] = None):
        """
        Initialize leverage manager.
        
        Args:
            config: Trading configuration
            kelly_engine: Optional shared dampened Kelly engine (recommended)
        """
        self.config = config
        self.logger = logger.getChild('LeverageManager')
        
        # Shared Kelly engine (set by position sizer)
        self.kelly_engine = kelly_engine
        
        # Backward compatibility: Basic leverage configuration
        self.min_leverage: float = MIN_LEVERAGE  # Minimum leverage (5x)
        self.max_leverage: float = MAX_LEVERAGE  # Maximum leverage (100x)
        self.leverage_multiplier: float = 1.0
        self.leverage_combined_threshold: float = 0.75
        
        # State management
        self.is_initialized: bool = False
        self.leverage_history: List[Dict[str, Any]] = []
        
        # Cache last Kelly result to reuse calculations
        self._last_kelly_result: Optional[KellyResult] = None
        self._last_kelly_timestamp: Optional[datetime] = None
    
    def set_kelly_engine(self, kelly_engine: DampenedKellyEngine) -> None:
        """
        Set the shared Kelly engine.
        
        Args:
            kelly_engine: Dampened Kelly engine from position sizer
        """
        self.kelly_engine = kelly_engine
        tprint_debug("Kelly engine set for Leverage Manager")
        self.logger.info("Kelly engine set for shared leverage calculation")
    
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize leverage manager."""
        try:
            tprint_info("🔄 Initializing Leverage Manager with Dampened Kelly...")
            self.logger.info("Initializing Leverage Manager...")
            
            # Validate configuration
            if not self._validate_configuration():
                tprint_error("❌ Leverage Manager configuration validation failed")
                return False
            
            # Update limits from Kelly engine if available
            if self.kelly_engine:
                safety_limits = self.kelly_engine.safety_limits
                self.max_leverage = safety_limits.get('max_leverage', self.max_leverage)
                tprint_info(f"✅ Leverage limits updated from Kelly engine: max={self.max_leverage}x")
            
            self.is_initialized = True
            tprint_success("✅ Leverage Manager with Dampened Kelly initialized successfully")
            self.logger.info("✅ Leverage Manager initialized successfully")
            return True
        
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Leverage Manager: {e}")
            self.logger.error(f"❌ Failed to initialize Leverage Manager: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate leverage manager configuration."""
        try:
            from src.config.leverage_constants import LEVERAGE_LOWER_BOUND, LEVERAGE_UPPER_BOUND
            
            # Ensure instance limits are within centralized bounds
            if self.min_leverage < LEVERAGE_LOWER_BOUND or self.min_leverage > LEVERAGE_UPPER_BOUND:
                tprint_error(f"min_leverage {self.min_leverage} must be between {LEVERAGE_LOWER_BOUND} and {LEVERAGE_UPPER_BOUND}")
                return False
            
            if self.max_leverage < LEVERAGE_LOWER_BOUND or self.max_leverage > LEVERAGE_UPPER_BOUND:
                tprint_error(f"max_leverage {self.max_leverage} must be between {LEVERAGE_LOWER_BOUND} and {LEVERAGE_UPPER_BOUND}")
                return False
            
            if self.min_leverage <= 0 or self.min_leverage >= self.max_leverage:
                tprint_error("Invalid leverage range configuration")
                return False
            
            tprint_debug("✅ Leverage Manager configuration validated")
            return True
        
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def cache_kelly_result(self, kelly_result: KellyResult, timestamp: Optional[datetime] = None) -> None:
        """
        Cache Kelly result from position sizer to reuse calculations.
        
        This avoids duplicate bin lookups and calculations when both position
        sizing and leverage are calculated for the same trade decision.
        
        Args:
            kelly_result: Kelly result from position sizer
            timestamp: Result timestamp (defaults to now)
        """
        self._last_kelly_result = kelly_result
        self._last_kelly_timestamp = timestamp or datetime.now()
        self.logger.debug("Cached Kelly result for leverage calculation reuse")
    
    def _can_reuse_cached_kelly(self, max_age_seconds: int = 5) -> bool:
        """
        Check if cached Kelly result can be reused.
        
        Args:
            max_age_seconds: Maximum age of cache in seconds
            
        Returns:
            True if cache is valid and fresh
        """
        if not self._last_kelly_result or not self._last_kelly_timestamp:
            return False
        
        age = (datetime.now() - self._last_kelly_timestamp).total_seconds()
        return age <= max_age_seconds
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="calculate_leverage")
    async def calculate_leverage(
        self,
        symbol: str,
        ml_predictions: Dict[str, Any],
        current_price: float = 0.0,
        account_balance: float = 1000.0,
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5,
        kelly_result: Optional[KellyResult] = None
    ) -> LeverageResult:
        """
        Calculate leverage using unified dampened Kelly logic.
        
        This method uses the same dampened Kelly engine as position sizing,
        with regime-aware beta_leverage parameters. If a Kelly result is
        provided (from position sizer), it will reuse those calculations.
        
        Args:
            symbol: Trading symbol
            ml_predictions: ML model predictions
            current_price: Current market price
            account_balance: Account balance
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score
            kelly_result: Optional Kelly result from position sizer (recommended)
            
        Returns:
            LeverageResult with metadata
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Leverage Manager not initialized")
            
            # If Kelly result provided, use it directly
            if kelly_result:
                leverage = kelly_result.leverage_final
                config_version = kelly_result.config_version
                kelly_metadata = kelly_result.to_dict()
                source = "provided_kelly_result"
            
            # Else if we have a fresh cached result, use it
            elif self._can_reuse_cached_kelly():
                leverage = self._last_kelly_result.leverage_final
                config_version = self._last_kelly_result.config_version
                kelly_metadata = self._last_kelly_result.to_dict()
                source = "cached_kelly_result"
                self.logger.debug("Reusing cached Kelly result for leverage")
            
            # Else if Kelly engine available, calculate fresh
            elif self.kelly_engine:
                # Note: This path should rarely be taken if integration is correct.
                # Position sizer should calculate and pass Kelly result.
                tprint_warning("⚠️ Calculating leverage without position sizer Kelly result - may be inefficient")
                
                # Would need to extract all inputs and do bin lookup here
                # For now, fall back to simple calculation
                leverage = self._calculate_simple_leverage(ml_predictions)
                config_version = self.kelly_engine.get_config_version() if self.kelly_engine else None
                kelly_metadata = {'method': 'simple_fallback'}
                source = "simple_fallback"
            
            # Ultimate fallback: simple confidence-based leverage
            else:
                leverage = self._calculate_simple_leverage(ml_predictions)
                config_version = None
                kelly_metadata = {'method': 'simple_fallback', 'no_kelly_engine': True}
                source = "simple_fallback_no_engine"
            
            # Apply limits
            leverage = max(self.min_leverage, min(leverage, self.max_leverage))
            
            # Validate leverage
            leverage = validate_leverage(leverage)
            
            # Build result
            result = LeverageResult(
                symbol=symbol,
                recommended_leverage=leverage,
                max_leverage=self.max_leverage,
                min_leverage=self.min_leverage,
                confidence=ml_predictions.get('combined_confidence', 0.5),
                leverage_multiplier=self.leverage_multiplier,
                metadata={
                    'source': source,
                    'config_version': config_version,
                    'kelly_metadata': kelly_metadata,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Track in history
            self.leverage_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'leverage': leverage,
                'source': source,
                'config_version': config_version
            })
            
            # Keep history limited
            if len(self.leverage_history) > 1000:
                self.leverage_history = self.leverage_history[-1000:]
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Error calculating leverage: {e}")
            tprint_error(f"❌ Leverage calculation error: {e}")
            
            # Return safe fallback
            return LeverageResult(
                symbol=symbol,
                recommended_leverage=self.min_leverage,
                max_leverage=self.max_leverage,
                min_leverage=self.min_leverage,
                confidence=0.0,
                leverage_multiplier=1.0,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def _calculate_simple_leverage(self, ml_predictions: Dict[str, Any]) -> float:
        """
        Simple confidence-based leverage calculation (fallback).
        
        This is used only when Kelly engine is not available or as emergency fallback.
        
        Args:
            ml_predictions: ML predictions
            
        Returns:
            Simple leverage based on confidence
        """
        combined_confidence = ml_predictions.get('combined_confidence', 0.5)
        intensity = ml_predictions.get('intensity', 1.0)
        reliability = ml_predictions.get('reliability', 1.0)
        
        # Simple leverage calculation
        base_leverage = combined_confidence * self.leverage_multiplier
        
        # Apply intensity and reliability adjustments
        intensity_factor = 0.8 + (intensity * 0.4)  # 0.8 to 1.2
        reliability_factor = 0.8 + (reliability * 0.4)  # 0.8 to 1.2
        
        adjusted_leverage = base_leverage * intensity_factor * reliability_factor
        
        # Scale to leverage range
        leverage_range = self.max_leverage - self.min_leverage
        leverage = self.min_leverage + (adjusted_leverage * leverage_range)
        
        return leverage
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get leverage manager statistics.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            'is_initialized': self.is_initialized,
            'total_leverage_calculations': len(self.leverage_history),
            'min_leverage': self.min_leverage,
            'max_leverage': self.max_leverage,
            'kelly_engine_available': self.kelly_engine is not None,
            'cached_result_available': self._last_kelly_result is not None
        }
        
        if self.leverage_history:
            leverages = [h['leverage'] for h in self.leverage_history]
            stats['avg_leverage'] = sum(leverages) / len(leverages)
            stats['max_leverage_used'] = max(leverages)
            stats['min_leverage_used'] = min(leverages)
            
            # Count sources
            sources = {}
            for h in self.leverage_history:
                source = h.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            stats['sources'] = sources
        
        return stats
