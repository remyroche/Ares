"""
Supervisor-Strategist Interface

This module defines the clear boundaries and interfaces between the Supervisor and Strategist components
to eliminate overlap and enforce separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class StrategyRequest:
    """Request structure for strategy generation."""
    market_data: pd.DataFrame
    current_price: float
    analysis_results: Optional[Dict[str, Any]] = None
    system_status: Optional[Dict[str, Any]] = None
    portfolio_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyResponse:
    """Response structure for strategy recommendations."""
    strategy_id: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    position_size: float
    risk_parameters: Dict[str, Any]
    reasoning: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class SystemStatusRequest:
    """Request structure for system status information."""
    component_name: str
    include_performance: bool = True
    include_health: bool = True
    include_configuration: bool = False


@dataclass
class SystemStatusResponse:
    """Response structure for system status information."""
    component_name: str
    is_healthy: bool
    performance_metrics: Dict[str, Any]
    health_status: Dict[str, Any]
    configuration: Optional[Dict[str, Any]] = None
    timestamp: datetime


class SupervisorStrategistInterface(ABC):
    """
    Abstract interface defining the boundaries between Supervisor and Strategist.
    
    This interface enforces clear separation of concerns:
    - Supervisor: System-level monitoring, coordination, and portfolio management
    - Strategist: Strategy generation, market analysis, and position sizing
    """

    @abstractmethod
    async def request_strategy_generation(self, request: StrategyRequest) -> StrategyResponse:
        """
        Request strategy generation from Strategist.
        
        This is the ONLY way Supervisor should interact with strategy generation.
        Supervisor provides market data and analysis results, Strategist returns strategy.
        
        Args:
            request: StrategyRequest containing market data and context
            
        Returns:
            StrategyResponse with strategy recommendations
        """
        pass

    @abstractmethod
    async def submit_strategy_performance(self, strategy_id: str, performance_metrics: Dict[str, Any]) -> bool:
        """
        Submit strategy performance metrics from Strategist to Supervisor.
        
        This allows Supervisor to track strategy performance for system-level decisions.
        
        Args:
            strategy_id: Unique identifier for the strategy
            performance_metrics: Performance data for the strategy
            
        Returns:
            bool: True if submission successful
        """
        pass

    @abstractmethod
    async def request_system_status(self, request: SystemStatusRequest) -> SystemStatusResponse:
        """
        Request system status information from Supervisor.
        
        This allows Strategist to get system context for strategy decisions.
        
        Args:
            request: SystemStatusRequest specifying what information is needed
            
        Returns:
            SystemStatusResponse with system status information
        """
        pass

    @abstractmethod
    async def notify_strategy_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Notify Supervisor of strategy-related events.
        
        This allows Strategist to inform Supervisor of important strategy events
        without Supervisor needing to monitor Strategist directly.
        
        Args:
            event_type: Type of event (e.g., "strategy_generated", "risk_triggered")
            event_data: Event-specific data
            
        Returns:
            bool: True if notification successful
        """
        pass


class SupervisorInterface(SupervisorStrategistInterface):
    """
    Supervisor-side implementation of the interface.
    
    Responsibilities:
    - System health monitoring
    - Component coordination
    - Portfolio-level risk management
    - Performance tracking
    - Recovery management
    """
    
    def __init__(self, supervisor_instance):
        self.supervisor = supervisor_instance
        self.logger = supervisor_instance.logger

    async def request_strategy_generation(self, request: StrategyRequest) -> StrategyResponse:
        """
        Supervisor requests strategy generation from Strategist.
        
        This method should delegate to the Strategist component without
        implementing strategy logic directly.
        """
        # Delegate to strategist component
        if hasattr(self.supervisor, 'components') and 'strategist' in self.supervisor.components:
            strategist = self.supervisor.components['strategist']
            if hasattr(strategist, 'generate_strategy'):
                strategy_result = await strategist.generate_strategy(
                    request.market_data,
                    request.current_price,
                    request.analysis_results
                )
                
                if strategy_result:
                    return StrategyResponse(
                        strategy_id=strategy_result.get('strategy_id', 'unknown'),
                        direction=strategy_result.get('direction', 'HOLD'),
                        confidence=strategy_result.get('confidence', 0.0),
                        position_size=strategy_result.get('position_size', 0.0),
                        risk_parameters=strategy_result.get('risk_parameters', {}),
                        reasoning=strategy_result.get('reasoning', []),
                        timestamp=datetime.now(),
                        metadata=strategy_result.get('metadata', {})
                    )
        
        # Fallback response if strategist is not available
        return StrategyResponse(
            strategy_id='fallback',
            direction='HOLD',
            confidence=0.0,
            position_size=0.0,
            risk_parameters={},
            reasoning=['Strategist not available'],
            timestamp=datetime.now(),
            metadata={'fallback': True}
        )

    async def submit_strategy_performance(self, strategy_id: str, performance_metrics: Dict[str, Any]) -> bool:
        """
        Receive strategy performance metrics from Strategist.
        
        This allows Supervisor to track strategy performance for system-level decisions
        without implementing strategy-specific logic.
        """
        try:
            # Store performance metrics for system-level analysis
            if not hasattr(self.supervisor, 'strategy_performance_history'):
                self.supervisor.strategy_performance_history = {}
            
            self.supervisor.strategy_performance_history[strategy_id] = {
                'metrics': performance_metrics,
                'timestamp': datetime.now()
            }
            
            # Update online learning if applicable
            if hasattr(self.supervisor, 'online_learning'):
                await self.supervisor.online_learning.update_model_performance(
                    strategy_id, 
                    performance_metrics.get('performance_score', 0.0)
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Error submitting strategy performance: {e}")
            return False

    async def request_system_status(self, request: SystemStatusRequest) -> SystemStatusResponse:
        """
        Provide system status information to Strategist.
        
        This gives Strategist access to system context without exposing
        internal supervisor implementation details.
        """
        try:
            component_name = request.component_name
            
            # Get component health status
            is_healthy = await self.supervisor._check_component_health(component_name)
            
            # Get performance metrics if requested
            performance_metrics = {}
            if request.include_performance:
                if hasattr(self.supervisor, 'supervision_results'):
                    performance_metrics = self.supervisor.supervision_results.get(component_name, {})
            
            # Get health status if requested
            health_status = {}
            if request.include_health:
                health_status = {
                    'is_healthy': is_healthy,
                    'last_check': datetime.now().isoformat(),
                    'recovery_attempts': self.supervisor.recovery_attempts.get(component_name, 0)
                }
            
            # Get configuration if requested
            configuration = None
            if request.include_configuration:
                if hasattr(self.supervisor, 'config'):
                    configuration = self.supervisor.config.get(component_name, {})
            
            return SystemStatusResponse(
                component_name=component_name,
                is_healthy=is_healthy,
                performance_metrics=performance_metrics,
                health_status=health_status,
                configuration=configuration,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error providing system status: {e}")
            return SystemStatusResponse(
                component_name=request.component_name,
                is_healthy=False,
                performance_metrics={},
                health_status={'error': str(e)},
                timestamp=datetime.now()
            )

    async def notify_strategy_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Receive strategy events from Strategist.
        
        This allows Supervisor to respond to strategy events without
        monitoring Strategist directly.
        """
        try:
            # Log the event
            self.logger.info(f"Strategy event received: {event_type} - {event_data}")
            
            # Handle specific event types
            if event_type == "risk_triggered":
                # Trigger portfolio-level risk management
                await self.supervisor._enforce_portfolio_guards()
            elif event_type == "strategy_generated":
                # Update supervision results
                if hasattr(self.supervisor, 'supervision_results'):
                    self.supervisor.supervision_results['strategist'] = event_data
            
            return True
        except Exception as e:
            self.logger.error(f"Error handling strategy event: {e}")
            return False


class StrategistInterface(SupervisorStrategistInterface):
    """
    Strategist-side implementation of the interface.
    
    Responsibilities:
    - Strategy generation
    - Market analysis integration
    - Strategy-specific risk management
    - Position sizing logic
    - Strategy history management
    """
    
    def __init__(self, strategist_instance):
        self.strategist = strategist_instance
        self.logger = strategist_instance.logger
        self.supervisor_interface = None

    def set_supervisor_interface(self, supervisor_interface):
        """Set the supervisor interface for communication."""
        self.supervisor_interface = supervisor_interface

    async def request_strategy_generation(self, request: StrategyRequest) -> StrategyResponse:
        """
        Generate strategy using Strategist's internal logic.
        
        This is the main strategy generation method that should be called
        by the Supervisor interface.
        """
        try:
            # Generate strategy using internal logic
            strategy_result = await self.strategist.generate_strategy(
                request.market_data,
                request.current_price,
                request.analysis_results
            )
            
            if strategy_result:
                # Create response
                response = StrategyResponse(
                    strategy_id=strategy_result.get('strategy_id', f"strategy_{datetime.now().timestamp()}"),
                    direction=strategy_result.get('direction', 'HOLD'),
                    confidence=strategy_result.get('confidence', 0.0),
                    position_size=strategy_result.get('position_size', 0.0),
                    risk_parameters=strategy_result.get('risk_parameters', {}),
                    reasoning=strategy_result.get('reasoning', []),
                    timestamp=datetime.now(),
                    metadata=strategy_result.get('metadata', {})
                )
                
                # Notify supervisor of strategy generation
                if self.supervisor_interface:
                    await self.supervisor_interface.notify_strategy_event(
                        "strategy_generated",
                        {
                            'strategy_id': response.strategy_id,
                            'direction': response.direction,
                            'confidence': response.confidence
                        }
                    )
                
                return response
            else:
                return StrategyResponse(
                    strategy_id='failed',
                    direction='HOLD',
                    confidence=0.0,
                    position_size=0.0,
                    risk_parameters={},
                    reasoning=['Strategy generation failed'],
                    timestamp=datetime.now(),
                    metadata={'error': 'Strategy generation failed'}
                )
        except Exception as e:
            self.logger.error(f"Error in strategy generation: {e}")
            return StrategyResponse(
                strategy_id='error',
                direction='HOLD',
                confidence=0.0,
                position_size=0.0,
                risk_parameters={},
                reasoning=[f'Error: {str(e)}'],
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )

    async def submit_strategy_performance(self, strategy_id: str, performance_metrics: Dict[str, Any]) -> bool:
        """
        Submit strategy performance to Supervisor.
        
        This method should be called internally by Strategist when
        performance data is available.
        """
        if self.supervisor_interface:
            return await self.supervisor_interface.submit_strategy_performance(strategy_id, performance_metrics)
        return False

    async def request_system_status(self, request: SystemStatusRequest) -> SystemStatusResponse:
        """
        Request system status from Supervisor.
        
        This allows Strategist to get system context for strategy decisions.
        """
        if self.supervisor_interface:
            return await self.supervisor_interface.request_system_status(request)
        else:
            return SystemStatusResponse(
                component_name=request.component_name,
                is_healthy=False,
                performance_metrics={},
                health_status={'error': 'No supervisor interface available'},
                timestamp=datetime.now()
            )

    async def notify_strategy_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Notify Supervisor of strategy events.
        
        This method should be called internally by Strategist when
        important events occur.
        """
        if self.supervisor_interface:
            return await self.supervisor_interface.notify_strategy_event(event_type, event_data)
        return False