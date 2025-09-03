"""
Recovery Manager Module.

This module handles automatic recovery and fallback mechanisms for
system components when failures occur.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, failed, warning


class RecoveryManager:
    """Manages automatic recovery and fallback mechanisms."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize recovery manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RecoveryManager")
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.last_recovery_attempt: Dict[str, float] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_recovery_attempts: int = config.get("max_recovery_attempts", 3)
        self.recovery_cooldown: int = config.get("recovery_cooldown", 300)  # 5 minutes
        self.max_history: int = config.get("max_recovery_history", 100)
        
        # Recovery strategies
        self.recovery_strategies = {
            "restart": self._restart_component,
            "reset": self._reset_component,
            "fallback": self._fallback_component,
            "isolate": self._isolate_component,
        }

    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
    )
    async def handle_component_failure(self, component_name: str, 
                                     error_details: Dict[str, Any]) -> bool:
        """
        Handle component failure with automatic recovery.
        
        Args:
            component_name: Name of the failed component
            error_details: Details about the failure
            
        Returns:
            True if recovery was successful
        """
        try:
            self.logger.warning(warning(f"Handling failure for component: {component_name}"))
            
            # Check if we should attempt recovery
            if not self._should_attempt_recovery(component_name):
                self.logger.error(failed(f"Max recovery attempts reached for {component_name}"))
                return False
            
            # Determine recovery strategy
            strategy = self._determine_recovery_strategy(component_name, error_details)
            
            # Execute recovery
            recovery_result = await self._execute_recovery(component_name, strategy, error_details)
            
            # Record recovery attempt
            self._record_recovery_attempt(component_name, strategy, recovery_result)
            
            return recovery_result["success"]
            
        except Exception as e:
            self.logger.error(error(f"Error in recovery process: {e}"))
            return False

    def _should_attempt_recovery(self, component_name: str) -> bool:
        """Check if recovery should be attempted."""
        # Check attempt count
        if self.recovery_attempts[component_name] >= self.max_recovery_attempts:
            return False
        
        # Check cooldown period
        last_attempt = self.last_recovery_attempt.get(component_name, 0)
        if time.time() - last_attempt < self.recovery_cooldown:
            return False
        
        return True

    def _determine_recovery_strategy(self, component_name: str, 
                                   error_details: Dict[str, Any]) -> str:
        """Determine the appropriate recovery strategy."""
        attempt_count = self.recovery_attempts[component_name]
        error_type = error_details.get("error_type", "unknown")
        
        # Progressive recovery strategies
        if attempt_count == 0:
            # First attempt: try restart
            return "restart"
        elif attempt_count == 1:
            # Second attempt: try reset
            return "reset"
        elif attempt_count == 2:
            # Third attempt: try fallback
            return "fallback"
        else:
            # Final attempt: isolate component
            return "isolate"

    async def _execute_recovery(self, component_name: str, 
                               strategy: str, 
                               error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the recovery strategy."""
        self.logger.info(f"Executing {strategy} recovery for {component_name}")
        
        result = {
            "component": component_name,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "message": "",
        }
        
        try:
            if strategy in self.recovery_strategies:
                recovery_func = self.recovery_strategies[strategy]
                success = await recovery_func(component_name, error_details)
                result["success"] = success
                result["message"] = f"{strategy} recovery {'succeeded' if success else 'failed'}"
            else:
                result["message"] = f"Unknown recovery strategy: {strategy}"
                
        except Exception as e:
            result["message"] = f"Recovery failed with error: {str(e)}"
            self.logger.error(error(f"Recovery execution failed: {e}"))
        
        return result

    async def _restart_component(self, component_name: str, 
                                error_details: Dict[str, Any]) -> bool:
        """Restart the component."""
        try:
            self.logger.info(f"Restarting component: {component_name}")
            # Implementation would restart the actual component
            # For now, simulate restart
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart component: {e}")
            return False

    async def _reset_component(self, component_name: str, 
                              error_details: Dict[str, Any]) -> bool:
        """Reset component to initial state."""
        try:
            self.logger.info(f"Resetting component: {component_name}")
            # Implementation would reset the actual component
            # For now, simulate reset
            await asyncio.sleep(2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset component: {e}")
            return False

    async def _fallback_component(self, component_name: str, 
                                 error_details: Dict[str, Any]) -> bool:
        """Switch to fallback implementation."""
        try:
            self.logger.info(f"Switching to fallback for component: {component_name}")
            # Implementation would switch to fallback component
            # For now, simulate fallback
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch to fallback: {e}")
            return False

    async def _isolate_component(self, component_name: str, 
                                error_details: Dict[str, Any]) -> bool:
        """Isolate the component to prevent further issues."""
        try:
            self.logger.warning(f"Isolating component: {component_name}")
            # Implementation would isolate the component
            # For now, simulate isolation
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            self.logger.error(f"Failed to isolate component: {e}")
            return False

    def _record_recovery_attempt(self, component_name: str, 
                               strategy: str, 
                               result: Dict[str, Any]) -> None:
        """Record recovery attempt details."""
        self.recovery_attempts[component_name] += 1
        self.last_recovery_attempt[component_name] = time.time()
        
        # Add to history
        self.recovery_history.append(result)
        if len(self.recovery_history) > self.max_history:
            self.recovery_history.pop(0)

    def reset_component_recovery(self, component_name: str) -> None:
        """Reset recovery attempts for a component."""
        self.recovery_attempts[component_name] = 0
        if component_name in self.last_recovery_attempt:
            del self.last_recovery_attempt[component_name]
        self.logger.info(f"Reset recovery attempts for {component_name}")

    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery status."""
        return {
            "recovery_attempts": dict(self.recovery_attempts),
            "components_in_recovery": list(self.last_recovery_attempt.keys()),
            "total_recovery_attempts": sum(self.recovery_attempts.values()),
            "recent_recoveries": self.recovery_history[-5:] if self.recovery_history else [],
        }

    def get_recovery_history(self, component_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recovery history, optionally filtered by component."""
        if component_name:
            return [r for r in self.recovery_history if r["component"] == component_name]
        return self.recovery_history.copy()