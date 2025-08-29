"""
Tactician Position Sizing Interface

This module defines the interface for position sizing, ensuring that only the Tactician
component handles position sizing logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class PositionSizingRequest:
    """Request structure for position sizing."""
    strategy_direction: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    market_data: pd.DataFrame
    risk_parameters: Dict[str, Any]
    portfolio_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PositionSizingResponse:
    """Response structure for position sizing recommendations."""
    position_size: float
    leverage: float
    entry_timing: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_metrics: Dict[str, Any] = None
    reasoning: list[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.risk_metrics is None:
            self.risk_metrics = {}
        if self.reasoning is None:
            self.reasoning = []


class TacticianPositionSizingInterface(ABC):
    """
    Abstract interface for position sizing, ensuring only Tactician handles this responsibility.
    
    This interface enforces that:
    - Only Tactician performs position sizing calculations
    - Supervisor and Strategist can request position sizing but cannot implement it
    - Position sizing includes leverage, timing, and risk management
    """

    @abstractmethod
    async def calculate_position_size(self, request: PositionSizingRequest) -> PositionSizingResponse:
        """
        Calculate position size based on strategy and market conditions.
        
        This is the ONLY method that should handle position sizing logic.
        
        Args:
            request: PositionSizingRequest containing strategy and market data
            
        Returns:
            PositionSizingResponse with position sizing recommendations
        """
        pass

    @abstractmethod
    async def validate_position_size(self, position_size: float, context: Dict[str, Any]) -> bool:
        """
        Validate position size against risk limits and portfolio constraints.
        
        Args:
            position_size: Proposed position size
            context: Portfolio and risk context
            
        Returns:
            bool: True if position size is valid
        """
        pass

    @abstractmethod
    async def get_position_sizing_history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """
        Get position sizing history for analysis and optimization.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            list[Dict[str, Any]]: Position sizing history
        """
        pass


class TacticianPositionSizingImplementation(TacticianPositionSizingInterface):
    """
    Tactician-side implementation of position sizing interface.
    
    This is the ONLY component that should implement position sizing logic.
    """
    
    def __init__(self, tactician_instance):
        self.tactician = tactician_instance
        self.logger = tactician_instance.logger

    async def calculate_position_size(self, request: PositionSizingRequest) -> PositionSizingResponse:
        """
        Calculate position size using Tactician's position sizing logic.
        
        This method delegates to the Tactician's position sizer component.
        """
        try:
            if not hasattr(self.tactician, 'position_sizer') or not self.tactician.position_sizer:
                self.logger.error("Position sizer not available")
                return self._create_fallback_response("Position sizer not available")

            # Delegate to position sizer
            position_sizer = self.tactician.position_sizer
            
            # Calculate position size using Tactician's logic
            position_size_result = await position_sizer.calculate_position_size_for_interface(
                confidence=request.confidence,
                direction=request.strategy_direction,
                current_price=request.current_price,
                market_data=request.market_data,
                risk_parameters=request.risk_parameters
            )

            # Create response
            response = PositionSizingResponse(
                position_size=position_size_result.get('position_size', 0.0),
                leverage=position_size_result.get('leverage', 1.0),
                entry_timing=position_size_result.get('entry_timing', 'immediate'),
                stop_loss=position_size_result.get('stop_loss'),
                take_profit=position_size_result.get('take_profit'),
                risk_metrics=position_size_result.get('risk_metrics', {}),
                reasoning=position_size_result.get('reasoning', []),
                timestamp=datetime.now()
            )

            # Validate position size
            is_valid = await self.validate_position_size(
                response.position_size, 
                {'portfolio_context': request.portfolio_context}
            )
            
            if not is_valid:
                response.position_size = 0.0
                response.reasoning.append("Position size invalidated by risk limits")

            return response

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self._create_fallback_response(f"Error: {str(e)}")

    async def validate_position_size(self, position_size: float, context: Dict[str, Any]) -> bool:
        """
        Validate position size against risk limits and portfolio constraints.
        """
        try:
            if not hasattr(self.tactician, 'position_sizer') or not self.tactician.position_sizer:
                return False

            # Delegate validation to position sizer
            position_sizer = self.tactician.position_sizer
            
            # Check against risk limits
            if hasattr(position_sizer, 'validate_position_size'):
                return await position_sizer.validate_position_size(position_size, context)
            
            # Basic validation
            if position_size <= 0 or position_size > 1.0:
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Error validating position size: {e}")
            return False

    async def get_position_sizing_history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """
        Get position sizing history from Tactician.
        """
        try:
            if not hasattr(self.tactician, 'position_sizer') or not self.tactician.position_sizer:
                return []

            position_sizer = self.tactician.position_sizer
            
            if hasattr(position_sizer, 'get_sizing_history'):
                history = await position_sizer.get_sizing_history()
                if limit:
                    history = history[-limit:]
                return history
            
            return []

        except Exception as e:
            self.logger.error(f"Error getting position sizing history: {e}")
            return []

    def _create_fallback_response(self, reason: str) -> PositionSizingResponse:
        """Create a fallback response when position sizing fails."""
        return PositionSizingResponse(
            position_size=0.0,
            leverage=1.0,
            entry_timing='hold',
            reasoning=[reason],
            timestamp=datetime.now()
        )


class SupervisorPositionSizingInterface(TacticianPositionSizingInterface):
    """
    Supervisor-side interface for requesting position sizing from Tactician.
    
    This interface allows Supervisor to request position sizing without implementing it.
    """
    
    def __init__(self, supervisor_instance):
        self.supervisor = supervisor_instance
        self.logger = supervisor_instance.logger
        self.tactician_interface = None

    def set_tactician_interface(self, tactician_interface):
        """Set the tactician interface for position sizing requests."""
        self.tactician_interface = tactician_interface

    async def calculate_position_size(self, request: PositionSizingRequest) -> PositionSizingResponse:
        """
        Request position sizing from Tactician.
        
        Supervisor delegates position sizing to Tactician without implementing it.
        """
        if self.tactician_interface:
            return await self.tactician_interface.calculate_position_size(request)
        else:
            self.logger.error("No tactician interface available for position sizing")
            return PositionSizingResponse(
                position_size=0.0,
                leverage=1.0,
                entry_timing='hold',
                reasoning=['No tactician interface available'],
                timestamp=datetime.now()
            )

    async def validate_position_size(self, position_size: float, context: Dict[str, Any]) -> bool:
        """
        Request position size validation from Tactician.
        """
        if self.tactician_interface:
            return await self.tactician_interface.validate_position_size(position_size, context)
        return False

    async def get_position_sizing_history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """
        Request position sizing history from Tactician.
        """
        if self.tactician_interface:
            return await self.tactician_interface.get_position_sizing_history(limit)
        return []


class StrategistPositionSizingInterface(TacticianPositionSizingInterface):
    """
    Strategist-side interface for requesting position sizing from Tactician.
    
    This interface allows Strategist to request position sizing without implementing it.
    """
    
    def __init__(self, strategist_instance):
        self.strategist = strategist_instance
        self.logger = strategist_instance.logger
        self.tactician_interface = None

    def set_tactician_interface(self, tactician_interface):
        """Set the tactician interface for position sizing requests."""
        self.tactician_interface = tactician_interface

    async def calculate_position_size(self, request: PositionSizingRequest) -> PositionSizingResponse:
        """
        Request position sizing from Tactician.
        
        Strategist delegates position sizing to Tactician without implementing it.
        """
        if self.tactician_interface:
            return await self.tactician_interface.calculate_position_size(request)
        else:
            self.logger.error("No tactician interface available for position sizing")
            return PositionSizingResponse(
                position_size=0.0,
                leverage=1.0,
                entry_timing='hold',
                reasoning=['No tactician interface available'],
                timestamp=datetime.now()
            )

    async def validate_position_size(self, position_size: float, context: Dict[str, Any]) -> bool:
        """
        Request position size validation from Tactician.
        """
        if self.tactician_interface:
            return await self.tactician_interface.validate_position_size(position_size, context)
        return False

    async def get_position_sizing_history(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """
        Request position sizing history from Tactician.
        """
        if self.tactician_interface:
            return await self.tactician_interface.get_position_sizing_history(limit)
        return []