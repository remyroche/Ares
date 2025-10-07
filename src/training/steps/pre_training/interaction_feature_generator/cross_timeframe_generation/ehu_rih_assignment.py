"""
EHU vs RIH Assignment Logic

Implements cost-aware, dynamic assignment of HTF features to:
- EHU (End-of-Hour Update): Features updated at HTF close, carried forward
- RIH (Real-time Incremental Update): Features updated incrementally with state
- Hybrid mode: Runtime switching based on latency headroom and market conditions
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum

from .staleness_curve import (
    StalenessCurve,
    StalenessCurveCalculator,
)


class UpdateStyle(Enum):
    """Update style for HTF features."""
    EHU = "ehu"  # End-of-Hour Update
    RIH = "rih"  # Real-time Incremental Update
    HYBRID = "hybrid"  # Dynamic switching


@dataclass
class AssignmentDecision:
    """Assignment decision for an HTF feature."""
    feature_name: str
    family: str
    lookback: int
    update_style: UpdateStyle
    expected_ic: float
    cost_per_ms: float
    staleness_curve: Dict[int, float]
    switch_conditions: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class CostBenefitAnalyzer:
    """Analyzes cost-benefit trade-offs for EHU vs RIH."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_cost_benefit(self, 
                           feature_name: str,
                           family: str,
                           lookback: int,
                           expected_ic: float,
                           staleness_curve: StalenessCurve) -> Dict[str, Any]:
        """
        Analyze cost-benefit trade-off for EHU vs RIH.
        
        Args:
            feature_name: Name of the feature
            family: Feature family
            lookback: Lookback period in minutes
            expected_ic: Expected IC of the feature
            staleness_curve: Staleness curve for the feature
            
        Returns:
            Cost-benefit analysis results
        """
        # Calculate costs
        ehu_cost = self._calculate_ehu_cost(family, lookback)
        rih_cost = self._calculate_rih_cost(family, lookback)
        
        # Calculate benefits (IC degradation due to staleness)
        ehu_benefit = self._calculate_ehu_benefit(expected_ic, staleness_curve)
        rih_benefit = self._calculate_rih_benefit(expected_ic, staleness_curve)
        
        # Calculate marginal benefit per cost
        ehu_marginal = (ehu_benefit - rih_benefit) / max(rih_cost - ehu_cost, 0.001)
        rih_marginal = (rih_benefit - ehu_benefit) / max(ehu_cost - rih_cost, 0.001)
        
        # Calculate cost per millisecond
        ehu_cost_per_ms = ehu_cost / max(lookback, 1)
        rih_cost_per_ms = rih_cost / max(lookback, 1)
        
        return {
            'ehu_cost': ehu_cost,
            'rih_cost': rih_cost,
            'ehu_benefit': ehu_benefit,
            'rih_benefit': rih_benefit,
            'ehu_marginal': ehu_marginal,
            'rih_marginal': rih_marginal,
            'ehu_cost_per_ms': ehu_cost_per_ms,
            'rih_cost_per_ms': rih_cost_per_ms,
            'net_benefit_ehu': ehu_benefit - ehu_cost,
            'net_benefit_rih': rih_benefit - rih_cost
        }
    
    def _calculate_ehu_cost(self, family: str, lookback: int) -> float:
        """Calculate EHU cost (typically very low)."""
        # EHU features are computed once per HTF period
        base_cost = 0.001  # ms per computation
        family_multiplier = self._get_family_multiplier(family)
        return base_cost * family_multiplier
    
    def _calculate_rih_cost(self, family: str, lookback: int) -> float:
        """Calculate RIH cost (incremental updates)."""
        # RIH features require incremental state maintenance
        base_cost = 0.01  # ms per update
        family_multiplier = self._get_family_multiplier(family)
        lookback_factor = np.log(lookback) / np.log(60)  # Scale with lookback
        return base_cost * family_multiplier * lookback_factor
    
    def _get_family_multiplier(self, family: str) -> float:
        """Get family-specific cost multiplier."""
        multipliers = {
            'trend_level_vol': 1.0,
            'oscillators': 1.2,
            'anchors': 0.8,
            'liquidity_micro': 1.1,
            'context': 1.3
        }
        return multipliers.get(family, 1.0)
    
    def _calculate_ehu_benefit(self, 
                             expected_ic: float,
                             staleness_curve: StalenessCurve) -> float:
        """Calculate EHU benefit (IC degradation due to staleness)."""
        # EHU features are stale for the entire HTF period
        # Use average staleness over the HTF period
        avg_staleness = staleness_curve.summary.average if staleness_curve.staleness_values else 0.5
        
        # IC degradation is proportional to staleness
        ic_degradation = expected_ic * avg_staleness
        return expected_ic - ic_degradation
    
    def _calculate_rih_benefit(self, 
                             expected_ic: float,
                             staleness_curve: StalenessCurve) -> float:
        """Calculate RIH benefit (minimal staleness)."""
        # RIH features have minimal staleness (just the base timeframe)
        min_staleness = staleness_curve.summary.at_base
        
        # IC degradation is minimal
        ic_degradation = expected_ic * min_staleness
        return expected_ic - ic_degradation


class HybridModeManager:
    """Manages hybrid mode switching between EHU and RIH."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Track switching history
        self.switch_history = []
        self.current_modes = {}  # feature_name -> UpdateStyle
    
    def should_switch_to_rih(self, 
                           feature_name: str,
                           current_mode: UpdateStyle,
                           latency_headroom: float,
                           market_conditions: Dict[str, Any]) -> bool:
        """
        Determine if feature should switch to RIH mode.
        
        Args:
            feature_name: Name of the feature
            current_mode: Current update mode
            latency_headroom: Available latency headroom in ms
            market_conditions: Current market conditions
            
        Returns:
            True if should switch to RIH
        """
        if current_mode == UpdateStyle.RIH:
            return False  # Already in RIH mode
        
        # Check latency headroom
        if latency_headroom < 10:  # Need at least 10ms headroom
            return False
        
        # Check market conditions
        volatility_level = market_conditions.get('volatility_level', 0.5)
        news_proximity = market_conditions.get('news_proximity', 0.0)
        
        # Switch to RIH in high volatility or near news
        if volatility_level > 0.7 or news_proximity > 0.5:
            return True
        
        # Check recent switching frequency (avoid thrashing)
        recent_switches = self._count_recent_switches(feature_name, minutes=30)
        if recent_switches > 2:  # Max 2 switches per 30 minutes
            return False
        
        return False
    
    def should_switch_to_ehu(self, 
                           feature_name: str,
                           current_mode: UpdateStyle,
                           latency_headroom: float,
                           market_conditions: Dict[str, Any]) -> bool:
        """
        Determine if feature should switch to EHU mode.
        
        Args:
            feature_name: Name of the feature
            current_mode: Current update mode
            latency_headroom: Available latency headroom in ms
            market_conditions: Current market conditions
            
        Returns:
            True if should switch to EHU
        """
        if current_mode == UpdateStyle.EHU:
            return False  # Already in EHU mode
        
        # Check latency headroom
        if latency_headroom < 5:  # Switch to EHU if low headroom
            return True
        
        # Check market conditions
        volatility_level = market_conditions.get('volatility_level', 0.5)
        news_proximity = market_conditions.get('news_proximity', 0.0)
        
        # Switch to EHU in low volatility and no news
        if volatility_level < 0.3 and news_proximity < 0.2:
            return True
        
        return False
    
    def _count_recent_switches(self, feature_name: str, minutes: int) -> int:
        """Count recent switches for a feature."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_switches = [
            switch for switch in self.switch_history
            if (switch['feature_name'] == feature_name and 
                switch['timestamp'] > cutoff_time)
        ]
        return len(recent_switches)
    
    def record_switch(self, 
                     feature_name: str,
                     from_mode: UpdateStyle,
                     to_mode: UpdateStyle,
                     reason: str):
        """Record a mode switch."""
        switch_record = {
            'feature_name': feature_name,
            'from_mode': from_mode.value,
            'to_mode': to_mode.value,
            'timestamp': datetime.now(),
            'reason': reason
        }
        self.switch_history.append(switch_record)
        self.current_modes[feature_name] = to_mode


class EHU_RIH_Assignment:
    """Main EHU/RIH assignment system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.staleness_calculator = StalenessCurveCalculator()
        self.cost_benefit_analyzer = CostBenefitAnalyzer(config)
        self.hybrid_manager = HybridModeManager(config)
    
    def assign_htf_features(self, 
                          phase2_results: Dict[str, Any],
                          sessionized_data: Dict[str, Any]) -> List[AssignmentDecision]:
        """
        Assign HTF features to EHU or RIH update styles.
        
        Args:
            phase2_results: Phase-2 optimization results
            sessionized_data: Sessionized data for context
            
        Returns:
            List of assignment decisions
        """
        self.logger.info("Starting EHU/RIH assignment")
        
        optimized_features = phase2_results.get('optimized_features', [])
        assignments = []
        
        for feature in optimized_features:
            try:
                # Calculate staleness curve
                staleness_curve = self.staleness_calculator.calculate_staleness_curve(
                    feature.feature_name,
                    feature.family,
                    feature.optimal_lookback
                )
                
                # Analyze cost-benefit
                cost_benefit = self.cost_benefit_analyzer.analyze_cost_benefit(
                    feature.feature_name,
                    feature.family,
                    feature.optimal_lookback,
                    feature.optimal_ic,
                    staleness_curve
                )
                
                # Make assignment decision
                assignment = self._make_assignment_decision(
                    feature, staleness_curve, cost_benefit
                )
                
                assignments.append(assignment)
                
            except Exception as e:
                self.logger.warning(f"Failed to assign {feature.feature_name}: {e}")
                continue
        
        self.logger.info(f"EHU/RIH assignment completed: {len(assignments)} features assigned")
        return assignments
    
    def _make_assignment_decision(self, 
                                feature,
                                staleness_curve: StalenessCurve,
                                cost_benefit: Dict[str, Any]) -> AssignmentDecision:
        """Make assignment decision for a single feature."""
        
        # Check if hybrid mode is enabled
        if self.config.hybrid_mode:
            # Start with EHU as default, allow switching
            initial_mode = UpdateStyle.EHU
            switch_conditions = self._create_switch_conditions(cost_benefit)
        else:
            # Choose based on cost-benefit analysis
            if cost_benefit['rih_marginal'] > self.config.rih_threshold:
                initial_mode = UpdateStyle.RIH
                switch_conditions = None
            else:
                initial_mode = UpdateStyle.EHU
                switch_conditions = None
        
        # Calculate cost per ms
        if initial_mode == UpdateStyle.RIH:
            cost_per_ms = cost_benefit['rih_cost_per_ms']
        else:
            cost_per_ms = cost_benefit['ehu_cost_per_ms']
        
        return AssignmentDecision(
            feature_name=feature.feature_name,
            family=feature.family,
            lookback=feature.optimal_lookback,
            update_style=initial_mode,
            expected_ic=feature.optimal_ic,
            cost_per_ms=cost_per_ms,
            staleness_curve=staleness_curve.staleness_values,
            switch_conditions=switch_conditions,
            metadata={
                'cost_benefit': cost_benefit,
                'staleness_curve_params': staleness_curve.curve_params
            }
        )
    
    def _create_switch_conditions(self, cost_benefit: Dict[str, Any]) -> Dict[str, Any]:
        """Create switch conditions for hybrid mode."""
        return {
            'switch_to_rih_threshold': self.config.rih_threshold,
            'latency_headroom_min': 10,  # ms
            'volatility_threshold': 0.7,
            'news_proximity_threshold': 0.5,
            'max_switches_per_hour': 4
        }
    
    def update_hybrid_assignments(self, 
                                current_assignments: List[AssignmentDecision],
                                latency_headroom: float,
                                market_conditions: Dict[str, Any]) -> List[AssignmentDecision]:
        """
        Update hybrid mode assignments based on current conditions.
        
        Args:
            current_assignments: Current assignment decisions
            latency_headroom: Available latency headroom in ms
            market_conditions: Current market conditions
            
        Returns:
            Updated assignment decisions
        """
        updated_assignments = []
        
        for assignment in current_assignments:
            if assignment.switch_conditions is None:
                # No switching allowed
                updated_assignments.append(assignment)
                continue
            
            current_mode = assignment.update_style
            
            # Check if should switch to RIH
            if self.hybrid_manager.should_switch_to_rih(
                assignment.feature_name, current_mode, latency_headroom, market_conditions
            ):
                # Switch to RIH
                new_assignment = AssignmentDecision(
                    feature_name=assignment.feature_name,
                    family=assignment.family,
                    lookback=assignment.lookback,
                    update_style=UpdateStyle.RIH,
                    expected_ic=assignment.expected_ic,
                    cost_per_ms=assignment.cost_per_ms * 2,  # RIH is more expensive
                    staleness_curve=assignment.staleness_curve,
                    switch_conditions=assignment.switch_conditions,
                    metadata=assignment.metadata
                )
                
                self.hybrid_manager.record_switch(
                    assignment.feature_name, current_mode, UpdateStyle.RIH, "High volatility/news"
                )
                
                updated_assignments.append(new_assignment)
            
            # Check if should switch to EHU
            elif self.hybrid_manager.should_switch_to_ehu(
                assignment.feature_name, current_mode, latency_headroom, market_conditions
            ):
                # Switch to EHU
                new_assignment = AssignmentDecision(
                    feature_name=assignment.feature_name,
                    family=assignment.family,
                    lookback=assignment.lookback,
                    update_style=UpdateStyle.EHU,
                    expected_ic=assignment.expected_ic,
                    cost_per_ms=assignment.cost_per_ms * 0.5,  # EHU is cheaper
                    staleness_curve=assignment.staleness_curve,
                    switch_conditions=assignment.switch_conditions,
                    metadata=assignment.metadata
                )
                
                self.hybrid_manager.record_switch(
                    assignment.feature_name, current_mode, UpdateStyle.EHU, "Low volatility/headroom"
                )
                
                updated_assignments.append(new_assignment)
            
            else:
                # No change
                updated_assignments.append(assignment)
        
        return updated_assignments
    
    def get_assignment_summary(self, assignments: List[AssignmentDecision]) -> Dict[str, Any]:
        """Get summary of assignment decisions."""
        ehu_count = sum(1 for a in assignments if a.update_style == UpdateStyle.EHU)
        rih_count = sum(1 for a in assignments if a.update_style == UpdateStyle.RIH)
        hybrid_count = sum(1 for a in assignments if a.update_style == UpdateStyle.HYBRID)
        
        total_cost = sum(a.cost_per_ms for a in assignments)
        avg_cost = total_cost / len(assignments) if assignments else 0
        
        return {
            'total_features': len(assignments),
            'ehu_count': ehu_count,
            'rih_count': rih_count,
            'hybrid_count': hybrid_count,
            'total_cost_ms': total_cost,
            'avg_cost_per_ms': avg_cost,
            'switch_history': self.hybrid_manager.switch_history[-10:]  # Last 10 switches
        }