"""
Trading Launcher with Hot Swap Functionality

Provides hot swap capability for trading parameters while trading is running:
- Kelly fraction
- Max loss per day (with 24-hour trading disable when threshold exceeded)
- Stop loss percentage
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from live_trading.trading_engine import TradingEngine
from live_trading.config import TradingConfig
from src.trading.sizing.position_sizer import PositionSizer
from src.trading.sizing.dampened_kelly_engine import DampenedKellyEngine, KellyConfigVersion
from src.utils.logger import system_logger

logger = system_logger.getChild("TradingLauncher")


class TradingParameterManager:
    """
    Manages hot swapping of trading parameters during live trading.
    
    Thread-safe parameter updates for:
    - Kelly fraction
    - Max daily loss
    - Stop loss percentage
    """
    
    def __init__(self, trading_engine: Optional[TradingEngine] = None):
        """
        Initialize parameter manager.
        
        Args:
            trading_engine: Optional TradingEngine instance to manage
        """
        self.trading_engine: Optional[TradingEngine] = trading_engine
        self.position_sizer: Optional[PositionSizer] = None
        self.dampened_kelly_engine: Optional[DampenedKellyEngine] = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Track daily loss disable state
        self._daily_loss_exceeded_at: Optional[datetime] = None
        self._trading_disabled_until: Optional[datetime] = None
        
        # Track current parameters
        self._current_kelly_fraction: Optional[float] = None
        self._current_max_daily_loss: Optional[float] = None
        self._current_stop_loss_pct: Optional[float] = None
        
        # Kelly sizing parameters (hot-swappable)
        self._current_max_leverage: Optional[float] = None
        self._current_max_per_trade_pct: Optional[float] = None
        self._current_max_exposure_per_asset: Optional[float] = None
        self._current_max_kelly_fraction: Optional[float] = None
        self._current_max_acceptable_drawdown: Optional[float] = None
        
        # Config version tracking for Kelly engine
        self._kelly_config_version_history: list = []
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_running = False
        
        logger.info("TradingParameterManager initialized")
    
    def set_trading_engine(self, trading_engine: TradingEngine) -> None:
        """Set the trading engine to manage."""
        with self._lock:
            self.trading_engine = trading_engine
            logger.info("Trading engine set for parameter manager")
            # Start monitoring if not already running
            if not self._monitoring_running:
                self._start_monitoring()
    
    def set_position_sizer(self, position_sizer: PositionSizer) -> None:
        """Set the position sizer to manage."""
        with self._lock:
            self.position_sizer = position_sizer
            logger.info("Position sizer set for parameter manager")
    
    def set_dampened_kelly_engine(self, kelly_engine: DampenedKellyEngine) -> None:
        """Set the dampened Kelly engine to manage."""
        with self._lock:
            self.dampened_kelly_engine = kelly_engine
            logger.info("Dampened Kelly engine set for parameter manager")
    
    def hot_swap_kelly_fraction(self, kelly_fraction: float) -> Dict[str, Any]:
        """
        Hot swap Kelly fraction for position sizing.
        
        Args:
            kelly_fraction: New Kelly fraction value (0.0 to 1.0)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                # Validate input
                if not 0.0 <= kelly_fraction <= 1.0:
                    raise ValueError(f"Kelly fraction must be between 0.0 and 1.0, got {kelly_fraction}")
                
                old_value = self._current_kelly_fraction
                
                # Update position sizer if available
                if self.position_sizer:
                    self.position_sizer.kelly_multiplier = kelly_fraction
                    self._current_kelly_fraction = kelly_fraction
                    
                    logger.info(f"✅ Hot swapped Kelly fraction: {old_value} -> {kelly_fraction}")
                    
                    return {
                        "success": True,
                        "parameter": "kelly_fraction",
                        "old_value": old_value,
                        "new_value": kelly_fraction,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Store for later if position sizer not available yet
                    self._current_kelly_fraction = kelly_fraction
                    
                    logger.warning(f"⚠️ Position sizer not available, stored Kelly fraction for later: {kelly_fraction}")
                    
                    return {
                        "success": True,
                        "parameter": "kelly_fraction",
                        "old_value": old_value,
                        "new_value": kelly_fraction,
                        "pending": True,
                        "message": "Position sizer not available, will apply when initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap Kelly fraction: {e}")
                return {
                    "success": False,
                    "parameter": "kelly_fraction",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    async def hot_swap_max_daily_loss(self, max_daily_loss: float) -> Dict[str, Any]:
        """
        Hot swap max daily loss threshold.
        
        If the current daily loss exceeds the new threshold, trading is disabled for 24 hours.
        
        Args:
            max_daily_loss: New max daily loss value (must be positive)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                # Validate input
                if max_daily_loss <= 0:
                    raise ValueError(f"Max daily loss must be positive, got {max_daily_loss}")
                
                old_value = self._current_max_daily_loss
                
                # Update trading config if available
                if self.trading_engine and self.trading_engine.config:
                    self.trading_engine.config.max_daily_loss = max_daily_loss
                    
                    # Update risk manager limits
                    if hasattr(self.trading_engine, 'risk_manager'):
                        self.trading_engine.risk_manager.risk_limits.max_daily_loss = max_daily_loss
                    
                    self._current_max_daily_loss = max_daily_loss
                    
                    # Check if we need to disable trading
                    should_disable = await self._check_and_disable_if_needed(max_daily_loss)
                    
                    logger.info(f"✅ Hot swapped max daily loss: {old_value} -> {max_daily_loss}")
                    if should_disable:
                        logger.warning(f"⚠️ Trading disabled for 24 hours due to daily loss threshold exceeded")
                    
                    return {
                        "success": True,
                        "parameter": "max_daily_loss",
                        "old_value": old_value,
                        "new_value": max_daily_loss,
                        "trading_disabled": should_disable,
                        "disabled_until": self._trading_disabled_until.isoformat() if self._trading_disabled_until else None,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Store for later if trading engine not available
                    self._current_max_daily_loss = max_daily_loss
                    
                    logger.warning(f"⚠️ Trading engine not available, stored max daily loss for later: {max_daily_loss}")
                    
                    return {
                        "success": True,
                        "parameter": "max_daily_loss",
                        "old_value": old_value,
                        "new_value": max_daily_loss,
                        "pending": True,
                        "message": "Trading engine not available, will apply when initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap max daily loss: {e}")
                return {
                    "success": False,
                    "parameter": "max_daily_loss",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _check_and_disable_if_needed(self, max_daily_loss: float) -> bool:
        """
        Check if daily loss exceeds threshold and disable trading if needed.
        
        Returns:
            True if trading was disabled, False otherwise
        """
        try:
            if not self.trading_engine or not hasattr(self.trading_engine, 'risk_manager'):
                return False
            
            # Get current daily PnL
            risk_summary = await self.trading_engine.risk_manager.get_risk_summary()
            total_daily_pnl = risk_summary.get("total_daily_pnl", 0.0)
            
            # Check if daily loss exceeds threshold
            if total_daily_pnl < -max_daily_loss:
                # Disable trading for 24 hours
                self._daily_loss_exceeded_at = datetime.now()
                self._trading_disabled_until = datetime.now() + timedelta(hours=24)
                
                # Pause trading
                await self.trading_engine.pause_trading()
                
                logger.warning(
                    f"⚠️ Daily loss threshold exceeded! "
                    f"Daily PnL: {total_daily_pnl:.2f}, Threshold: {max_daily_loss:.2f}. "
                    f"Trading disabled until {self._trading_disabled_until.isoformat()}"
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking daily loss: {e}")
            return False
    
    def hot_swap_stop_loss_percentage(self, stop_loss_pct: float) -> Dict[str, Any]:
        """
        Hot swap stop loss percentage.
        
        Args:
            stop_loss_pct: New stop loss percentage (must be positive, e.g., 2.0 for 2%)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                # Validate input
                if stop_loss_pct <= 0:
                    raise ValueError(f"Stop loss percentage must be positive, got {stop_loss_pct}")
                
                old_value = self._current_stop_loss_pct
                
                # Update trading config if available
                if self.trading_engine and self.trading_engine.config:
                    self.trading_engine.config.stop_loss_percentage = stop_loss_pct
                    self._current_stop_loss_pct = stop_loss_pct
                    
                    logger.info(f"✅ Hot swapped stop loss percentage: {old_value} -> {stop_loss_pct}%")
                    
                    return {
                        "success": True,
                        "parameter": "stop_loss_percentage",
                        "old_value": old_value,
                        "new_value": stop_loss_pct,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Store for later if trading engine not available
                    self._current_stop_loss_pct = stop_loss_pct
                    
                    logger.warning(f"⚠️ Trading engine not available, stored stop loss percentage for later: {stop_loss_pct}%")
                    
                    return {
                        "success": True,
                        "parameter": "stop_loss_percentage",
                        "old_value": old_value,
                        "new_value": stop_loss_pct,
                        "pending": True,
                        "message": "Trading engine not available, will apply when initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap stop loss percentage: {e}")
                return {
                    "success": False,
                    "parameter": "stop_loss_percentage",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    async def check_trading_disable_status(self) -> Dict[str, Any]:
        """
        Check if trading is disabled and when it will be re-enabled.
        
        Returns:
            Dict with disable status information
        """
        with self._lock:
            try:
                # Check if 24-hour disable period has passed
                if self._trading_disabled_until:
                    if datetime.now() >= self._trading_disabled_until:
                        # Re-enable trading
                        if self.trading_engine:
                            await self.trading_engine.resume_trading()
                        
                        logger.info("✅ 24-hour trading disable period ended, trading re-enabled")
                        
                        disabled_until = self._trading_disabled_until.isoformat()
                        self._trading_disabled_until = None
                        self._daily_loss_exceeded_at = None
                        
                        return {
                            "disabled": False,
                            "was_disabled_until": disabled_until,
                            "now_enabled": True,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # Still disabled
                        return {
                            "disabled": True,
                            "disabled_until": self._trading_disabled_until.isoformat(),
                            "disabled_at": self._daily_loss_exceeded_at.isoformat() if self._daily_loss_exceeded_at else None,
                            "time_remaining_hours": (self._trading_disabled_until - datetime.now()).total_seconds() / 3600,
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Not disabled
                return {
                    "disabled": False,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"❌ Error checking trading disable status: {e}")
                return {
                    "disabled": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current values of all managed parameters.
        
        Returns:
            Dict with current parameter values
        """
        with self._lock:
            return {
                "kelly_fraction": self._current_kelly_fraction,
                "max_daily_loss": self._current_max_daily_loss,
                "stop_loss_percentage": self._current_stop_loss_pct,
                "trading_disabled_until": self._trading_disabled_until.isoformat() if self._trading_disabled_until else None,
                "daily_loss_exceeded_at": self._daily_loss_exceeded_at.isoformat() if self._daily_loss_exceeded_at else None,
                "timestamp": datetime.now().isoformat()
            }
    
    def _start_monitoring(self) -> None:
        """Start monitoring task for daily loss checks."""
        if self._monitoring_running:
            return
        
        self._monitoring_running = True
        
        # Create task if we're in an async context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._monitoring_task = asyncio.create_task(self._monitor_daily_loss())
            else:
                # If no event loop, we'll create it when needed
                pass
        except RuntimeError:
            # No event loop, will be created when needed
            pass
        
        logger.info("Started daily loss monitoring")
    
    async def _monitor_daily_loss(self) -> None:
        """Continuously monitor daily loss and disable trading if threshold exceeded."""
        while self._monitoring_running:
            try:
                # Check if trading should be re-enabled
                await self.check_trading_disable_status()
                
                # Check if daily loss exceeds threshold
                if self.trading_engine and self._current_max_daily_loss:
                    risk_summary = await self.trading_engine.risk_manager.get_risk_summary()
                    total_daily_pnl = risk_summary.get("total_daily_pnl", 0.0)
                    
                    # Only check if trading is not already disabled
                    if not self._trading_disabled_until and total_daily_pnl < -self._current_max_daily_loss:
                        # Disable trading for 24 hours
                        await self._check_and_disable_if_needed(self._current_max_daily_loss)
                
                # Wait before next check (check every minute)
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in daily loss monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        self._monitoring_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Stopped daily loss monitoring")
    
    # ========================================
    # Kelly Sizing Hot-Swap Methods
    # ========================================
    
    def _update_kelly_engine_config(self, param_name: str, param_value: Any) -> int:
        """
        Update a single parameter in the Kelly engine config.
        
        Args:
            param_name: Name of the parameter to update
            param_value: New value for the parameter
            
        Returns:
            New config version number
        """
        if not self.dampened_kelly_engine:
            raise RuntimeError("Dampened Kelly engine not set")
        
        # Get current config
        current_config = self.dampened_kelly_engine.config.copy()
        
        # Update safety limits
        if 'safety_limits' not in current_config:
            current_config['safety_limits'] = {}
        
        current_config['safety_limits'][param_name] = param_value
        
        # Update engine with new config
        new_version = self.dampened_kelly_engine.update_config(current_config)
        
        # Track in history
        self._kelly_config_version_history.append({
            'version': new_version,
            'timestamp': datetime.now().isoformat(),
            'parameter': param_name,
            'value': param_value
        })
        
        return new_version
    
    def hot_swap_max_leverage(self, leverage: float) -> Dict[str, Any]:
        """
        Hot swap maximum leverage limit.
        
        Args:
            leverage: New maximum leverage (e.g., 3.0 for 3x)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                # Validate input
                if leverage <= 0:
                    raise ValueError(f"Leverage must be positive, got {leverage}")
                if leverage > 100:
                    raise ValueError(f"Leverage must be <= 100x, got {leverage}")
                
                old_value = self._current_max_leverage
                
                # Update Kelly engine if available
                if self.dampened_kelly_engine:
                    new_version = self._update_kelly_engine_config('max_leverage', leverage)
                    self._current_max_leverage = leverage
                    
                    logger.info(f"✅ Hot swapped max leverage: {old_value} -> {leverage}x (version {new_version})")
                    
                    return {
                        "success": True,
                        "parameter": "max_leverage",
                        "old_value": old_value,
                        "new_value": leverage,
                        "config_version": new_version,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._current_max_leverage = leverage
                    logger.warning(f"⚠️ Kelly engine not available, stored max leverage for later: {leverage}x")
                    
                    return {
                        "success": True,
                        "parameter": "max_leverage",
                        "old_value": old_value,
                        "new_value": leverage,
                        "pending": True,
                        "message": "Kelly engine not available, will apply when initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap max leverage: {e}")
                return {
                    "success": False,
                    "parameter": "max_leverage",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    def hot_swap_max_per_trade_pct(self, pct: float) -> Dict[str, Any]:
        """
        Hot swap maximum position size per trade.
        
        Args:
            pct: New maximum percentage (e.g., 0.15 for 15%)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                # Validate input
                if not 0.0 < pct <= 1.0:
                    raise ValueError(f"Percentage must be between 0 and 1, got {pct}")
                
                old_value = self._current_max_per_trade_pct
                
                if self.dampened_kelly_engine:
                    new_version = self._update_kelly_engine_config('max_per_trade_pct', pct)
                    self._current_max_per_trade_pct = pct
                    
                    logger.info(f"✅ Hot swapped max per trade: {old_value} -> {pct*100:.1f}% (version {new_version})")
                    
                    return {
                        "success": True,
                        "parameter": "max_per_trade_pct",
                        "old_value": old_value,
                        "new_value": pct,
                        "config_version": new_version,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._current_max_per_trade_pct = pct
                    logger.warning(f"⚠️ Kelly engine not available, stored max per trade for later: {pct*100:.1f}%")
                    
                    return {
                        "success": True,
                        "parameter": "max_per_trade_pct",
                        "old_value": old_value,
                        "new_value": pct,
                        "pending": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap max per trade: {e}")
                return {"success": False, "parameter": "max_per_trade_pct", "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def hot_swap_max_exposure_per_asset(self, pct: float) -> Dict[str, Any]:
        """
        Hot swap maximum exposure per asset.
        
        Args:
            pct: New maximum exposure (e.g., 0.30 for 30%)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                if not 0.0 < pct <= 1.0:
                    raise ValueError(f"Percentage must be between 0 and 1, got {pct}")
                
                old_value = self._current_max_exposure_per_asset
                
                if self.dampened_kelly_engine:
                    new_version = self._update_kelly_engine_config('max_exposure_per_asset', pct)
                    self._current_max_exposure_per_asset = pct
                    
                    logger.info(f"✅ Hot swapped max exposure per asset: {old_value} -> {pct*100:.1f}% (version {new_version})")
                    
                    return {
                        "success": True,
                        "parameter": "max_exposure_per_asset",
                        "old_value": old_value,
                        "new_value": pct,
                        "config_version": new_version,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._current_max_exposure_per_asset = pct
                    return {"success": True, "parameter": "max_exposure_per_asset", "old_value": old_value, "new_value": pct, "pending": True, "timestamp": datetime.now().isoformat()}
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap max exposure per asset: {e}")
                return {"success": False, "parameter": "max_exposure_per_asset", "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def hot_swap_max_kelly_fraction(self, fraction: float) -> Dict[str, Any]:
        """
        Hot swap maximum Kelly fraction (never exceed this fraction of theoretical Kelly).
        
        Args:
            fraction: New maximum Kelly fraction (e.g., 0.5 for half Kelly)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                if not 0.0 < fraction <= 1.0:
                    raise ValueError(f"Fraction must be between 0 and 1, got {fraction}")
                
                old_value = self._current_max_kelly_fraction
                
                if self.dampened_kelly_engine:
                    new_version = self._update_kelly_engine_config('max_kelly_fraction', fraction)
                    self._current_max_kelly_fraction = fraction
                    
                    logger.info(f"✅ Hot swapped max Kelly fraction: {old_value} -> {fraction} (version {new_version})")
                    
                    return {
                        "success": True,
                        "parameter": "max_kelly_fraction",
                        "old_value": old_value,
                        "new_value": fraction,
                        "config_version": new_version,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._current_max_kelly_fraction = fraction
                    return {"success": True, "parameter": "max_kelly_fraction", "old_value": old_value, "new_value": fraction, "pending": True, "timestamp": datetime.now().isoformat()}
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap max Kelly fraction: {e}")
                return {"success": False, "parameter": "max_kelly_fraction", "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def hot_swap_max_acceptable_drawdown(self, dd: float) -> Dict[str, Any]:
        """
        Hot swap maximum acceptable drawdown threshold.
        
        Args:
            dd: New maximum acceptable drawdown (e.g., 0.15 for 15%)
            
        Returns:
            Dict with status information
        """
        with self._lock:
            try:
                if not 0.0 < dd <= 1.0:
                    raise ValueError(f"Drawdown must be between 0 and 1, got {dd}")
                
                old_value = self._current_max_acceptable_drawdown
                
                if self.dampened_kelly_engine:
                    new_version = self._update_kelly_engine_config('max_acceptable_drawdown', dd)
                    self._current_max_acceptable_drawdown = dd
                    
                    logger.info(f"✅ Hot swapped max acceptable drawdown: {old_value} -> {dd*100:.1f}% (version {new_version})")
                    
                    return {
                        "success": True,
                        "parameter": "max_acceptable_drawdown",
                        "old_value": old_value,
                        "new_value": dd,
                        "config_version": new_version,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._current_max_acceptable_drawdown = dd
                    return {"success": True, "parameter": "max_acceptable_drawdown", "old_value": old_value, "new_value": dd, "pending": True, "timestamp": datetime.now().isoformat()}
                    
            except Exception as e:
                logger.error(f"❌ Failed to hot swap max acceptable drawdown: {e}")
                return {"success": False, "parameter": "max_acceptable_drawdown", "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_kelly_config_version_history(self) -> List[Dict[str, Any]]:
        """
        Get history of Kelly config version updates.
        
        Returns:
            List of config version history entries
        """
        with self._lock:
            return self._kelly_config_version_history.copy()


# Global instance for easy access
_parameter_manager: Optional[TradingParameterManager] = None


def get_parameter_manager(trading_engine: Optional[TradingEngine] = None) -> TradingParameterManager:
    """
    Get or create the global parameter manager instance.
    
    Args:
        trading_engine: Optional TradingEngine to set if manager doesn't exist
        
    Returns:
        TradingParameterManager instance
    """
    global _parameter_manager
    
    if _parameter_manager is None:
        _parameter_manager = TradingParameterManager(trading_engine)
    elif trading_engine and not _parameter_manager.trading_engine:
        _parameter_manager.set_trading_engine(trading_engine)
    
    return _parameter_manager


def hot_swap_kelly_fraction(kelly_fraction: float) -> Dict[str, Any]:
    """
    Convenience function to hot swap Kelly fraction.
    
    Args:
        kelly_fraction: New Kelly fraction value (0.0 to 1.0)
        
    Returns:
        Dict with status information
    """
    manager = get_parameter_manager()
    return manager.hot_swap_kelly_fraction(kelly_fraction)


async def hot_swap_max_daily_loss(max_daily_loss: float) -> Dict[str, Any]:
    """
    Convenience function to hot swap max daily loss.
    
    Args:
        max_daily_loss: New max daily loss value
        
    Returns:
        Dict with status information
    """
    manager = get_parameter_manager()
    return await manager.hot_swap_max_daily_loss(max_daily_loss)


def hot_swap_stop_loss_percentage(stop_loss_pct: float) -> Dict[str, Any]:
    """
    Convenience function to hot swap stop loss percentage.
    
    Args:
        stop_loss_pct: New stop loss percentage
        
    Returns:
        Dict with status information
    """
    manager = get_parameter_manager()
    return manager.hot_swap_stop_loss_percentage(stop_loss_pct)


# Example usage:
"""
# Example 1: Hot swap Kelly fraction (non-blocking)
from src.launcher.trading_launcher import hot_swap_kelly_fraction

result = hot_swap_kelly_fraction(0.3)  # Set Kelly fraction to 30%
print(result)
# {'success': True, 'parameter': 'kelly_fraction', 'old_value': None, 'new_value': 0.3, ...}


# Example 2: Hot swap max daily loss (async)
import asyncio
from src.launcher.trading_launcher import hot_swap_max_daily_loss

async def update_daily_loss():
    result = await hot_swap_max_daily_loss(200.0)  # Set max daily loss to $200
    print(result)
    # If daily loss exceeds threshold, trading will be automatically disabled for 24 hours

asyncio.run(update_daily_loss())


# Example 3: Hot swap stop loss percentage
from src.launcher.trading_launcher import hot_swap_stop_loss_percentage

result = hot_swap_stop_loss_percentage(5.0)  # Set stop loss to 5%
print(result)
# {'success': True, 'parameter': 'stop_loss_percentage', 'old_value': None, 'new_value': 5.0, ...}


# Example 4: Get current parameters
from src.launcher.trading_launcher import get_parameter_manager

manager = get_parameter_manager()
current_params = manager.get_current_parameters()
print(current_params)


# Example 5: Check trading disable status
import asyncio
from src.launcher.trading_launcher import get_parameter_manager

async def check_status():
    manager = get_parameter_manager()
    status = await manager.check_trading_disable_status()
    if status.get('disabled'):
        print(f"Trading disabled until {status['disabled_until']}")
    else:
        print("Trading is enabled")

asyncio.run(check_status())
"""
