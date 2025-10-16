"""
Unified Search Space Evolution Interface for NAS and TAS Systems

This module provides a unified interface for dynamic search space evolution that can be used
by both NAS and TAS systems to adapt their search spaces based on performance feedback,
market conditions, and regime changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from .unified_architecture_config import ArchitectureType, OptimizationObjective
from ..core.dynamic_search_space import (
    DynamicSearchSpace, DynamicSearchSpaceConfig, SearchSpaceEvolutionStrategy,
    MarketCondition, MarketConditionAnalysis
)

logger = logging.getLogger(__name__)

class EvolutionTrigger(Enum):
    """Triggers for search space evolution."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    REGIME_CHANGE = "regime_change"
    MARKET_CONDITION_CHANGE = "market_condition_change"
    TIME_BASED = "time_based"
    MANUAL = "manual"
    CONVERGENCE_STAGNATION = "convergence_stagnation"

class EvolutionAction(Enum):
    """Actions to take during search space evolution."""
    EXPAND_SEARCH_SPACE = "expand_search_space"
    CONTRACT_SEARCH_SPACE = "contract_search_space"
    MUTATE_PARAMETERS = "mutate_parameters"
    ADD_NEW_OPERATIONS = "add_new_operations"
    REMOVE_POOR_PERFORMERS = "remove_poor_performers"
    ADJUST_CONSTRAINTS = "adjust_constraints"
    RESET_SEARCH_SPACE = "reset_search_space"

@dataclass
class EvolutionEvent:
    """Represents a search space evolution event."""
    event_id: str
    trigger: EvolutionTrigger
    action: EvolutionAction
    timestamp: datetime
    performance_context: Dict[str, Any]
    market_context: Dict[str, Any]
    evolution_details: Dict[str, Any]
    success: bool = False
    performance_impact: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedEvolutionConfig:
    """Configuration for unified search space evolution."""

    # Evolution triggers
    enable_performance_based_evolution: bool = True
    enable_regime_based_evolution: bool = True
    enable_market_based_evolution: bool = True
    enable_time_based_evolution: bool = False

    # Performance thresholds
    performance_degradation_threshold: float = 0.05
    performance_improvement_threshold: float = 0.03
    convergence_stagnation_threshold: int = 50  # Number of evaluations without improvement

    # Evolution frequency
    min_evolution_interval: int = 100  # Minimum evaluations between evolutions
    max_evolution_interval: int = 500  # Maximum evaluations between evolutions
    time_based_interval_hours: int = 24  # Hours for time-based evolution

    # Evolution intensity
    evolution_intensity: float = 0.3  # How aggressive the evolution should be
    max_parameter_change: float = 0.5  # Maximum change to any parameter

    # Market condition thresholds
    volatility_change_threshold: float = 0.1
    trend_change_threshold: float = 0.2
    regime_stability_threshold: float = 0.7

    # Search space constraints
    min_search_space_size: int = 10
    max_search_space_size: int = 1000
    preserve_top_performers: float = 0.2  # Preserve top X% of architectures

    # Validation
    enable_evolution_validation: bool = True
    validation_window: int = 50  # Number of evaluations to validate evolution

@dataclass
class EvolutionResult:
    """Result from search space evolution."""
    evolution_event: EvolutionEvent
    search_space_changes: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    evolution_time: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedSearchSpaceEvolutionManager:
    """Unified search space evolution manager for NAS and TAS systems."""

    def __init__(self,
                 config: UnifiedEvolutionConfig,
                 nas_search_space: Any = None,
                 tas_search_space: Any = None,
                 market_data: Optional[pd.DataFrame] = None):
        """Initialize the unified evolution manager.

        Args:
            config: Evolution configuration
            nas_search_space: NAS search space to evolve
            tas_search_space: TAS search space to evolve
            market_data: Market data for regime analysis
        """
        self.config = config
        self.nas_search_space = nas_search_space
        self.tas_search_space = tas_search_space
        self.market_data = market_data

        self.logger = logging.getLogger(self.__class__.__name__)

        # Evolution state
        self.evolution_history: List[EvolutionEvent] = []
        self.performance_history: deque = deque(maxlen=1000)
        self.market_history: deque = deque(maxlen=1000)
        self.last_evolution_time = None
        self.evaluation_count = 0

        # Dynamic search spaces
        self.dynamic_nas_space = None
        self.dynamic_tas_space = None

        # Initialize dynamic search spaces
        self._initialize_dynamic_search_spaces()

        self.logger.info("✅ Unified Search Space Evolution Manager initialized")
        self.logger.info(f"   NAS Search Space: {'✅' if nas_search_space else '❌'}")
        self.logger.info(f"   TAS Search Space: {'✅' if tas_search_space else '❌'}")
        self.logger.info(f"   Market Data: {'✅' if market_data is not None else '❌'}")

    def _initialize_dynamic_search_spaces(self):
        """Initialize dynamic search spaces for both NAS and TAS."""
        try:
            # Create dynamic search space configuration
            dynamic_config = DynamicSearchSpaceConfig(
                evolution_strategy=SearchSpaceEvolutionStrategy.ADAPTIVE,
                evolution_frequency=self.config.min_evolution_interval,
                performance_window=self.config.validation_window
            )

            # Initialize dynamic search spaces
            if self.nas_search_space:
                self.dynamic_nas_space = DynamicSearchSpace(
                    config=dynamic_config,
                    architecture_type="neural"
                )
                self.logger.info("✅ Dynamic NAS search space initialized")

            if self.tas_search_space:
                self.dynamic_tas_space = DynamicSearchSpace(
                    config=dynamic_config,
                    architecture_type="tree"
                )
                self.logger.info("✅ Dynamic TAS search space initialized")

        except Exception as e:
            self.logger.error(f"❌ Dynamic search space initialization failed: {e}")
            raise

    def update_performance(self,
                         architecture_type: str,
                         performance_metrics: Dict[str, float],
                         architecture_info: Dict[str, Any] = None):
        """Update performance metrics for evolution tracking."""
        try:
            # Store performance data
            performance_data = {
                'timestamp': datetime.now(),
                'architecture_type': architecture_type,
                'performance_metrics': performance_metrics,
                'architecture_info': architecture_info or {}
            }

            self.performance_history.append(performance_data)
            self.evaluation_count += 1

            # Check if evolution is needed
            if self._should_evolve():
                self._trigger_evolution(EvolutionTrigger.PERFORMANCE_DEGRADATION)

        except Exception as e:
            self.logger.error(f"❌ Performance update failed: {e}")

    def update_market_conditions(self, market_analysis: MarketConditionAnalysis):
        """Update market conditions for regime-based evolution."""
        try:
            # Store market data
            market_data = {
                'timestamp': datetime.now(),
                'market_analysis': market_analysis
            }

            self.market_history.append(market_data)

            # Check if market-based evolution is needed
            if self._should_evolve_based_on_market(market_analysis):
                self._trigger_evolution(EvolutionTrigger.MARKET_CONDITION_CHANGE)

        except Exception as e:
            self.logger.error(f"❌ Market conditions update failed: {e}")

    def _should_evolve(self) -> bool:
        """Check if search space should evolve based on performance."""
        try:
            # Check minimum interval
            if (self.last_evolution_time and
                self.evaluation_count - self._get_last_evolution_evaluation_count() < self.config.min_evolution_interval):
                return False

            # Check maximum interval
            if (self.last_evolution_time and
                self.evaluation_count - self._get_last_evolution_evaluation_count() > self.config.max_evolution_interval):
                return True

            # Check performance degradation
            if self.config.enable_performance_based_evolution:
                if self._has_performance_degraded():
                    return True

            # Check convergence stagnation
            if self._has_convergence_stagnated():
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Evolution check failed: {e}")
            return False

    def _should_evolve_based_on_market(self, market_analysis: MarketConditionAnalysis) -> bool:
        """Check if search space should evolve based on market conditions."""
        try:
            if not self.config.enable_market_based_evolution:
                return False

            # Check minimum interval
            if (self.last_evolution_time and
                self.evaluation_count - self._get_last_evolution_evaluation_count() < self.config.min_evolution_interval):
                return False

            # Check market condition changes
            if len(self.market_history) < 2:
                return False

            previous_analysis = self.market_history[-2]['market_analysis']

            # Check volatility change
            volatility_change = abs(market_analysis.volatility - previous_analysis.volatility)
            if volatility_change > self.config.volatility_change_threshold:
                return True

            # Check trend change
            trend_change = abs(market_analysis.trend_strength - previous_analysis.trend_strength)
            if trend_change > self.config.trend_change_threshold:
                return True

            # Check regime stability
            if market_analysis.regime_stability < self.config.regime_stability_threshold:
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Market-based evolution check failed: {e}")
            return False

    def _has_performance_degraded(self) -> bool:
        """Check if performance has degraded significantly."""
        try:
            if len(self.performance_history) < 20:
                return False

            # Get recent performance
            recent_performances = [p['performance_metrics'].get('score', 0.0)
                                 for p in list(self.performance_history)[-20:]]

            if len(recent_performances) < 10:
                return False

            # Compare recent vs previous performance
            recent_avg = np.mean(recent_performances[-10:])
            previous_avg = np.mean(recent_performances[-20:-10])

            degradation = (previous_avg - recent_avg) / previous_avg if previous_avg > 0 else 0

            return degradation > self.config.performance_degradation_threshold

        except Exception as e:
            self.logger.warning(f"Performance degradation check failed: {e}")
            return False

    def _has_convergence_stagnated(self) -> bool:
        """Check if convergence has stagnated."""
        try:
            if len(self.performance_history) < self.config.convergence_stagnation_threshold:
                return False

            # Get recent performance
            recent_performances = [p['performance_metrics'].get('score', 0.0)
                                 for p in list(self.performance_history)[-self.config.convergence_stagnation_threshold:]]

            if len(recent_performances) < self.config.convergence_stagnation_threshold:
                return False

            # Check if performance has been stagnant
            max_performance = max(recent_performances)
            min_performance = min(recent_performances)
            performance_range = max_performance - min_performance

            # If range is very small, consider it stagnant
            return performance_range < 0.01

        except Exception as e:
            self.logger.warning(f"Convergence stagnation check failed: {e}")
            return False

    def _trigger_evolution(self, trigger: EvolutionTrigger):
        """Trigger search space evolution."""
        try:
            self.logger.info(f"🔄 Triggering search space evolution: {trigger.value}")

            # Create evolution event
            event_id = f"evolution_{len(self.evolution_history)}_{int(time.time())}"
            evolution_event = EvolutionEvent(
                event_id=event_id,
                trigger=trigger,
                action=EvolutionAction.MUTATE_PARAMETERS,  # Default action
                timestamp=datetime.now(),
                performance_context=self._get_performance_context(),
                market_context=self._get_market_context(),
                evolution_details={}
            )

            # Determine evolution action based on trigger
            action = self._determine_evolution_action(trigger, evolution_event)
            evolution_event.action = action

            # Perform evolution
            evolution_result = self._perform_evolution(evolution_event)

            # Update state
            self.last_evolution_time = datetime.now()
            self.evolution_history.append(evolution_event)

            self.logger.info(f"✅ Evolution completed: {evolution_result.success}")

        except Exception as e:
            self.logger.error(f"❌ Evolution trigger failed: {e}")

    def _determine_evolution_action(self, trigger: EvolutionTrigger, event: EvolutionEvent) -> EvolutionAction:
        """Determine the appropriate evolution action based on trigger and context."""
        try:
            if trigger == EvolutionTrigger.PERFORMANCE_DEGRADATION:
                # If performance degraded, try expanding search space
                return EvolutionAction.EXPAND_SEARCH_SPACE

            elif trigger == EvolutionTrigger.REGIME_CHANGE:
                # If regime changed, add new operations
                return EvolutionAction.ADD_NEW_OPERATIONS

            elif trigger == EvolutionTrigger.MARKET_CONDITION_CHANGE:
                # If market conditions changed, adjust parameters
                return EvolutionAction.MUTATE_PARAMETERS

            elif trigger == EvolutionTrigger.CONVERGENCE_STAGNATION:
                # If convergence stagnated, remove poor performers
                return EvolutionAction.REMOVE_POOR_PERFORMERS

            elif trigger == EvolutionTrigger.TIME_BASED:
                # Time-based evolution, adjust constraints
                return EvolutionAction.ADJUST_CONSTRAINTS

            else:
                # Default action
                return EvolutionAction.MUTATE_PARAMETERS

        except Exception as e:
            self.logger.warning(f"Evolution action determination failed: {e}")
            return EvolutionAction.MUTATE_PARAMETERS

    def _perform_evolution(self, event: EvolutionEvent) -> EvolutionResult:
        """Perform the actual search space evolution."""
        start_time = time.time()

        try:
            # Get performance before evolution
            performance_before = self._get_current_performance()

            # Perform evolution based on action
            search_space_changes = {}

            if event.action == EvolutionAction.EXPAND_SEARCH_SPACE:
                search_space_changes = self._expand_search_space()

            elif event.action == EvolutionAction.CONTRACT_SEARCH_SPACE:
                search_space_changes = self._contract_search_space()

            elif event.action == EvolutionAction.MUTATE_PARAMETERS:
                search_space_changes = self._mutate_parameters()

            elif event.action == EvolutionAction.ADD_NEW_OPERATIONS:
                search_space_changes = self._add_new_operations()

            elif event.action == EvolutionAction.REMOVE_POOR_PERFORMERS:
                search_space_changes = self._remove_poor_performers()

            elif event.action == EvolutionAction.ADJUST_CONSTRAINTS:
                search_space_changes = self._adjust_constraints()

            elif event.action == EvolutionAction.RESET_SEARCH_SPACE:
                search_space_changes = self._reset_search_space()

            # Update evolution details
            event.evolution_details = search_space_changes

            # Get performance after evolution
            performance_after = self._get_current_performance()

            # Calculate performance impact
            performance_impact = self._calculate_performance_impact(performance_before, performance_after)
            event.performance_impact = performance_impact

            # Determine success
            success = self._evaluate_evolution_success(performance_impact)
            event.success = success

            evolution_time = time.time() - start_time

            result = EvolutionResult(
                evolution_event=event,
                search_space_changes=search_space_changes,
                performance_before=performance_before,
                performance_after=performance_after,
                evolution_time=evolution_time,
                success=success,
                metadata={
                    'trigger': event.trigger.value,
                    'action': event.action.value,
                    'evolution_intensity': self.config.evolution_intensity
                }
            )

            return result

        except Exception as e:
            evolution_time = time.time() - start_time
            self.logger.error(f"❌ Evolution performance failed: {e}")

            return EvolutionResult(
                evolution_event=event,
                search_space_changes={},
                performance_before={},
                performance_after={},
                evolution_time=evolution_time,
                success=False,
                metadata={'error': str(e)}
            )

    def _expand_search_space(self) -> Dict[str, Any]:
        """Expand the search space by adding new parameters or ranges."""
        changes = {}

        try:
            # Expand NAS search space
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.expand_search_space(
                    intensity=self.config.evolution_intensity
                )
                changes['nas_space'] = nas_changes

            # Expand TAS search space
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.expand_search_space(
                    intensity=self.config.evolution_intensity
                )
                changes['tas_space'] = tas_changes

            self.logger.info("✅ Search space expanded")

        except Exception as e:
            self.logger.error(f"❌ Search space expansion failed: {e}")
            changes['error'] = str(e)

        return changes

    def _contract_search_space(self) -> Dict[str, Any]:
        """Contract the search space by removing poor-performing regions."""
        changes = {}

        try:
            # Contract NAS search space
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.contract_search_space(
                    intensity=self.config.evolution_intensity
                )
                changes['nas_space'] = nas_changes

            # Contract TAS search space
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.contract_search_space(
                    intensity=self.config.evolution_intensity
                )
                changes['tas_space'] = tas_changes

            self.logger.info("✅ Search space contracted")

        except Exception as e:
            self.logger.error(f"❌ Search space contraction failed: {e}")
            changes['error'] = str(e)

        return changes

    def _mutate_parameters(self) -> Dict[str, Any]:
        """Mutate search space parameters."""
        changes = {}

        try:
            # Mutate NAS parameters
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.mutate_parameters(
                    intensity=self.config.evolution_intensity,
                    max_change=self.config.max_parameter_change
                )
                changes['nas_space'] = nas_changes

            # Mutate TAS parameters
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.mutate_parameters(
                    intensity=self.config.evolution_intensity,
                    max_change=self.config.max_parameter_change
                )
                changes['tas_space'] = tas_changes

            self.logger.info("✅ Search space parameters mutated")

        except Exception as e:
            self.logger.error(f"❌ Parameter mutation failed: {e}")
            changes['error'] = str(e)

        return changes

    def _add_new_operations(self) -> Dict[str, Any]:
        """Add new operations to the search space."""
        changes = {}

        try:
            # Add new NAS operations
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.add_new_operations(
                    intensity=self.config.evolution_intensity
                )
                changes['nas_space'] = nas_changes

            # Add new TAS operations
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.add_new_operations(
                    intensity=self.config.evolution_intensity
                )
                changes['tas_space'] = tas_changes

            self.logger.info("✅ New operations added to search space")

        except Exception as e:
            self.logger.error(f"❌ Adding new operations failed: {e}")
            changes['error'] = str(e)

        return changes

    def _remove_poor_performers(self) -> Dict[str, Any]:
        """Remove poor-performing regions from search space."""
        changes = {}

        try:
            # Remove poor NAS performers
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.remove_poor_performers(
                    preserve_ratio=self.config.preserve_top_performers
                )
                changes['nas_space'] = nas_changes

            # Remove poor TAS performers
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.remove_poor_performers(
                    preserve_ratio=self.config.preserve_top_performers
                )
                changes['tas_space'] = tas_changes

            self.logger.info("✅ Poor performers removed from search space")

        except Exception as e:
            self.logger.error(f"❌ Removing poor performers failed: {e}")
            changes['error'] = str(e)

        return changes

    def _adjust_constraints(self) -> Dict[str, Any]:
        """Adjust search space constraints."""
        changes = {}

        try:
            # Adjust NAS constraints
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.adjust_constraints(
                    intensity=self.config.evolution_intensity
                )
                changes['nas_space'] = nas_changes

            # Adjust TAS constraints
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.adjust_constraints(
                    intensity=self.config.evolution_intensity
                )
                changes['tas_space'] = tas_changes

            self.logger.info("✅ Search space constraints adjusted")

        except Exception as e:
            self.logger.error(f"❌ Constraint adjustment failed: {e}")
            changes['error'] = str(e)

        return changes

    def _reset_search_space(self) -> Dict[str, Any]:
        """Reset search space to initial state."""
        changes = {}

        try:
            # Reset NAS search space
            if self.dynamic_nas_space:
                nas_changes = self.dynamic_nas_space.reset_to_initial_state()
                changes['nas_space'] = nas_changes

            # Reset TAS search space
            if self.dynamic_tas_space:
                tas_changes = self.dynamic_tas_space.reset_to_initial_state()
                changes['tas_space'] = tas_changes

            self.logger.info("✅ Search space reset to initial state")

        except Exception as e:
            self.logger.error(f"❌ Search space reset failed: {e}")
            changes['error'] = str(e)

        return changes

    def _get_performance_context(self) -> Dict[str, Any]:
        """Get current performance context."""
        try:
            if not self.performance_history:
                return {}

            recent_performances = [p['performance_metrics'].get('score', 0.0)
                                 for p in list(self.performance_history)[-20:]]

            return {
                'recent_performance': recent_performances,
                'average_performance': np.mean(recent_performances) if recent_performances else 0.0,
                'max_performance': max(recent_performances) if recent_performances else 0.0,
                'min_performance': min(recent_performances) if recent_performances else 0.0,
                'performance_trend': self._calculate_performance_trend(recent_performances),
                'evaluation_count': self.evaluation_count
            }

        except Exception as e:
            self.logger.warning(f"Performance context retrieval failed: {e}")
            return {}

    def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context."""
        try:
            if not self.market_history:
                return {}

            latest_market = self.market_history[-1]['market_analysis']

            return {
                'volatility': latest_market.volatility,
                'trend_strength': latest_market.trend_strength,
                'regime_stability': latest_market.regime_stability,
                'market_efficiency': latest_market.market_efficiency,
                'risk_level': latest_market.risk_level,
                'condition': latest_market.condition.value,
                'confidence': latest_market.confidence
            }

        except Exception as e:
            self.logger.warning(f"Market context retrieval failed: {e}")
            return {}

    def _get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        try:
            if not self.performance_history:
                return {'score': 0.0}

            latest_performance = self.performance_history[-1]['performance_metrics']
            return latest_performance

        except Exception as e:
            self.logger.warning(f"Current performance retrieval failed: {e}")
            return {'score': 0.0}

    def _calculate_performance_impact(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate the impact of evolution on performance."""
        try:
            before_score = before.get('score', 0.0)
            after_score = after.get('score', 0.0)

            if before_score == 0:
                return 0.0

            impact = (after_score - before_score) / before_score
            return impact

        except Exception as e:
            self.logger.warning(f"Performance impact calculation failed: {e}")
            return 0.0

    def _evaluate_evolution_success(self, performance_impact: float) -> bool:
        """Evaluate if evolution was successful."""
        return performance_impact >= 0.0  # Success if performance didn't decrease

    def _calculate_performance_trend(self, performances: List[float]) -> float:
        """Calculate performance trend (positive = improving)."""
        try:
            if len(performances) < 2:
                return 0.0

            # Simple linear trend
            x = np.arange(len(performances))
            y = np.array(performances)

            if np.var(x) == 0:
                return 0.0

            slope = np.cov(x, y)[0, 1] / np.var(x)
            return slope

        except Exception as e:
            self.logger.warning(f"Performance trend calculation failed: {e}")
            return 0.0

    def _get_last_evolution_evaluation_count(self) -> int:
        """Get evaluation count at last evolution."""
        try:
            if not self.evolution_history:
                return 0

            last_evolution = self.evolution_history[-1]
            return last_evolution.metadata.get('evaluation_count', 0)

        except Exception as e:
            self.logger.warning(f"Last evolution evaluation count retrieval failed: {e}")
            return 0

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        try:
            if not self.evolution_history:
                return {}

            successful_evolutions = [e for e in self.evolution_history if e.success]

            trigger_counts = defaultdict(int)
            action_counts = defaultdict(int)
            performance_impacts = [e.performance_impact for e in self.evolution_history]

            for event in self.evolution_history:
                trigger_counts[event.trigger.value] += 1
                action_counts[event.action.value] += 1

            return {
                'total_evolutions': len(self.evolution_history),
                'successful_evolutions': len(successful_evolutions),
                'success_rate': len(successful_evolutions) / len(self.evolution_history) if self.evolution_history else 0.0,
                'average_performance_impact': np.mean(performance_impacts) if performance_impacts else 0.0,
                'trigger_distribution': dict(trigger_counts),
                'action_distribution': dict(action_counts),
                'last_evolution_time': self.last_evolution_time.isoformat() if self.last_evolution_time else None,
                'total_evaluations': self.evaluation_count
            }

        except Exception as e:
            self.logger.warning(f"Evolution statistics retrieval failed: {e}")
            return {}

    def manual_evolution(self, action: EvolutionAction) -> EvolutionResult:
        """Trigger manual evolution."""
        try:
            self.logger.info(f"🔧 Manual evolution triggered: {action.value}")

            # Create manual evolution event
            event_id = f"manual_evolution_{len(self.evolution_history)}_{int(time.time())}"
            evolution_event = EvolutionEvent(
                event_id=event_id,
                trigger=EvolutionTrigger.MANUAL,
                action=action,
                timestamp=datetime.now(),
                performance_context=self._get_performance_context(),
                market_context=self._get_market_context(),
                evolution_details={}
            )

            # Perform evolution
            evolution_result = self._perform_evolution(evolution_event)

            # Update state
            self.last_evolution_time = datetime.now()
            self.evolution_history.append(evolution_event)

            self.logger.info(f"✅ Manual evolution completed: {evolution_result.success}")

            return evolution_result

        except Exception as e:
            self.logger.error(f"❌ Manual evolution failed: {e}")

            # Return error result
            return EvolutionResult(
                evolution_event=EvolutionEvent(
                    event_id="error",
                    trigger=EvolutionTrigger.MANUAL,
                    action=action,
                    timestamp=datetime.now(),
                    performance_context={},
                    market_context={},
                    evolution_details={'error': str(e)}
                ),
                search_space_changes={},
                performance_before={},
                performance_after={},
                evolution_time=0.0,
                success=False,
                metadata={'error': str(e)}
            )

def create_unified_evolution_manager(
    nas_search_space: Any = None,
    tas_search_space: Any = None,
    market_data: Optional[pd.DataFrame] = None,
    config: UnifiedEvolutionConfig = None
) -> UnifiedSearchSpaceEvolutionManager:
    """Create a unified search space evolution manager instance."""
    if config is None:
        config = UnifiedEvolutionConfig()

    return UnifiedSearchSpaceEvolutionManager(
        config=config,
        nas_search_space=nas_search_space,
        tas_search_space=tas_search_space,
        market_data=market_data
    )

def quick_evolution_setup(
    nas_search_space: Any = None,
    tas_search_space: Any = None,
    enable_performance_evolution: bool = True,
    enable_regime_evolution: bool = True
) -> UnifiedSearchSpaceEvolutionManager:
    """Quick evolution manager setup with default settings."""
    config = UnifiedEvolutionConfig(
        enable_performance_based_evolution=enable_performance_evolution,
        enable_regime_based_evolution=enable_regime_evolution,
        evolution_intensity=0.3,
        min_evolution_interval=100
    )

    return create_unified_evolution_manager(
        nas_search_space=nas_search_space,
        tas_search_space=tas_search_space,
        config=config
    )
