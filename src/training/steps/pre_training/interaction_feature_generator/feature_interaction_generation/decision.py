"""
Decision Logic for Discrete vs Blended Lookback Selection with Hysteresis

This module implements the decision logic for choosing between discrete lookback
choices and blended approaches, incorporating hysteresis and simplicity priors
to ensure stable, production-ready lookback selections.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import configuration and previous stage results
from .config import LookbackOptimizationConfig, FamilyType, HysteresisConfig, ExportConfig
from .ic_surface import ICSurfaceResult
from .wf_stability import StabilityResult
from .hierarchical import HierarchicalResult

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of lookback decisions."""
    DISCRETE = "discrete"
    BLEND = "blend"
    DEFAULT = "default"
    INACTIVE = "inactive"


@dataclass
class LookbackSpec:
    """Specification for lookback selection."""
    decision_type: DecisionType
    primary_lookback: Optional[float] = None
    secondary_lookback: Optional[float] = None
    blend_weights: Optional[Tuple[float, float]] = None
    effective_lookback: Optional[float] = None
    confidence_score: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'decision_type': self.decision_type.value,
            'primary_lookback': self.primary_lookback,
            'secondary_lookback': self.secondary_lookback,
            'blend_weights': self.blend_weights,
            'effective_lookback': self.effective_lookback,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning
        }


@dataclass
class DecisionResult:
    """Result of lookback decision process."""
    family: FamilyType
    symbol: str
    lookback_spec: LookbackSpec
    previous_lookback: Optional[float] = None
    change_magnitude: float = 0.0
    expected_ic_gain: float = 0.0
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'family': self.family.value,
            'symbol': self.symbol,
            'lookback_spec': self.lookback_spec.to_dict(),
            'previous_lookback': self.previous_lookback,
            'change_magnitude': self.change_magnitude,
            'expected_ic_gain': self.expected_ic_gain,
            'execution_time': self.execution_time
        }


class HysteresisManager:
    """Manages hysteresis logic for lookback stability."""
    
    def __init__(self, config: HysteresisConfig):
        self.config = config
        self.previous_lookbacks: Dict[Tuple[str, FamilyType], float] = {}
    
    def should_change_lookback(self, symbol: str, family: FamilyType, 
                              new_lookback: float, expected_ic_gain: float) -> bool:
        """Determine if lookback should be changed based on hysteresis rules."""
        key = (symbol, family)
        previous_lookback = self.previous_lookbacks.get(key)
        
        if previous_lookback is None:
            # First time, allow change
            return True
        
        # Check magnitude of change
        log_change = abs(np.log(new_lookback) - np.log(previous_lookback))
        if log_change < self.config.min_delta_log_l:
            return False
        
        # Check IC gain threshold
        if expected_ic_gain < self.config.min_delta_ic_sigma:
            return False
        
        return True
    
    def update_lookback(self, symbol: str, family: FamilyType, lookback: float) -> None:
        """Update stored lookback for hysteresis tracking."""
        key = (symbol, family)
        self.previous_lookbacks[key] = lookback
    
    def get_previous_lookback(self, symbol: str, family: FamilyType) -> Optional[float]:
        """Get previous lookback for a symbol-family combination."""
        key = (symbol, family)
        return self.previous_lookbacks.get(key)


class BlendOptimizer:
    """Optimizes blend weights for multi-window approaches."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_blend_weights(self, data: pd.DataFrame, target: np.ndarray,
                              family: FamilyType, feature_name: str,
                              lookback1: float, lookback2: float) -> Tuple[float, float]:
        """Optimize blend weights for two lookback windows."""
        try:
            # Generate features for both lookbacks
            feature1 = self._generate_feature(data, family, feature_name, int(lookback1))
            feature2 = self._generate_feature(data, family, feature_name, int(lookback2))
            
            # Remove NaN values
            valid_mask = np.isfinite(feature1) & np.isfinite(feature2) & np.isfinite(target)
            if np.sum(valid_mask) < 10:
                return 0.5, 0.5  # Equal weights as fallback
            
            feature1_clean = feature1[valid_mask]
            feature2_clean = feature2[valid_mask]
            target_clean = target[valid_mask]
            
            # Optimize weights using linear regression
            X = np.column_stack([feature1_clean, feature2_clean])
            y = target_clean
            
            # Add L2 penalty for regularization
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Extract weights and normalize
            w1, w2 = reg.coef_
            
            # Ensure non-negative weights
            w1 = max(0.0, w1)
            w2 = max(0.0, w2)
            
            # Normalize to sum to 1
            total_weight = w1 + w2
            if total_weight > 0:
                w1_norm = w1 / total_weight
                w2_norm = w2 / total_weight
            else:
                w1_norm = 0.5
                w2_norm = 0.5
            
            return w1_norm, w2_norm
            
        except Exception as e:
            self.logger.warning(f"Blend optimization failed: {e}. Using equal weights.")
            return 0.5, 0.5
    
    def _generate_feature(self, data: pd.DataFrame, family: FamilyType, 
                         feature_name: str, lookback: int) -> np.ndarray:
        """Generate feature for given lookback (simplified implementation)."""
        # This is a placeholder - in practice, this would call the actual feature generation
        if family == FamilyType.MOMENTUM and 'close' in data.columns:
            return data['close'].pct_change(lookback).values
        elif family == FamilyType.VOLATILITY and 'close' in data.columns:
            returns = data['close'].pct_change()
            alpha = 2 / (lookback + 1)
            ew_var = returns.ewm(alpha=alpha).var()
            return np.sqrt(ew_var).values
        else:
            return np.zeros(len(data))


class LookbackDecisionMaker:
    """Main class for making lookback decisions."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.hysteresis_manager = HysteresisManager(config.hysteresis)
        self.blend_optimizer = BlendOptimizer(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def make_decision(self, symbol: str, family: FamilyType,
                     ic_surface_result: ICSurfaceResult,
                     stability_result: StabilityResult,
                     hierarchical_result: Optional[HierarchicalResult] = None,
                     data: Optional[pd.DataFrame] = None,
                     target: Optional[np.ndarray] = None,
                     feature_name: str = "") -> DecisionResult:
        """Make lookback decision for a single symbol-family combination."""
        start_time = time.time()
        
        try:
            tprint_info(f"Making lookback decision for {symbol}-{family.value}...")
            
            # Get previous lookback for hysteresis
            previous_lookback = self.hysteresis_manager.get_previous_lookback(symbol, family)
            
            # Determine decision type
            decision_type = self._determine_decision_type(
                ic_surface_result, stability_result, hierarchical_result
            )
            
            # Create lookback specification
            lookback_spec = self._create_lookback_spec(
                decision_type, ic_surface_result, stability_result, 
                hierarchical_result, data, target, feature_name
            )
            
            # Apply hysteresis
            if previous_lookback is not None:
                should_change = self._apply_hysteresis(
                    symbol, family, lookback_spec, previous_lookback, ic_surface_result
                )
                
                if not should_change:
                    # Keep previous lookback
                    lookback_spec = LookbackSpec(
                        decision_type=DecisionType.DISCRETE,
                        primary_lookback=previous_lookback,
                        effective_lookback=previous_lookback,
                        confidence_score=0.5,
                        reasoning="Hysteresis: keeping previous lookback"
                    )
            
            # Calculate change metrics
            change_magnitude = 0.0
            if previous_lookback is not None and lookback_spec.effective_lookback is not None:
                change_magnitude = abs(np.log(lookback_spec.effective_lookback) - 
                                     np.log(previous_lookback))
            
            # Calculate expected IC gain
            expected_ic_gain = self._calculate_expected_ic_gain(
                ic_surface_result, lookback_spec, previous_lookback
            )
            
            # Update hysteresis tracking
            if lookback_spec.effective_lookback is not None:
                self.hysteresis_manager.update_lookback(symbol, family, lookback_spec.effective_lookback)
            
            execution_time = time.time() - start_time
            
            result = DecisionResult(
                family=family,
                symbol=symbol,
                lookback_spec=lookback_spec,
                previous_lookback=previous_lookback,
                change_magnitude=change_magnitude,
                expected_ic_gain=expected_ic_gain,
                execution_time=execution_time
            )
            
            tprint_info(f"Decision: {decision_type.value}, Lookback: {lookback_spec.effective_lookback}")
            tprint_info(f"Change magnitude: {change_magnitude:.3f}, IC gain: {expected_ic_gain:.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Lookback decision failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return default decision
            return DecisionResult(
                family=family,
                symbol=symbol,
                lookback_spec=LookbackSpec(
                    decision_type=DecisionType.DEFAULT,
                    primary_lookback=self._get_default_lookback(family),
                    effective_lookback=self._get_default_lookback(family),
                    confidence_score=0.0,
                    reasoning=f"Error occurred: {str(e)}"
                ),
                execution_time=execution_time
            )
    
    def _determine_decision_type(self, ic_surface_result: ICSurfaceResult,
                                stability_result: StabilityResult,
                                hierarchical_result: Optional[HierarchicalResult]) -> DecisionType:
        """Determine the type of lookback decision."""
        # Check if family should be inactive
        if (ic_surface_result.optimal_ic < 0.01 or 
            stability_result.stability_score < 0.3 or
            ic_surface_result.r_squared < 0.1):
            return DecisionType.INACTIVE
        
        # Check if blend is recommended
        if (stability_result.recommendation == "blend_recommended" or
            stability_result.match_rate < self.config.hysteresis.min_fold_match_rate or
            (hierarchical_result and 
             any(hdi_upper - hdi_lower > self.config.hysteresis.max_hdi_width 
                 for hdi_lower, hdi_upper in zip(hierarchical_result.family_hdi_lower.values(),
                                                hierarchical_result.family_hdi_upper.values())))):
            return DecisionType.BLEND
        
        # Default to discrete choice
        return DecisionType.DISCRETE
    
    def _create_lookback_spec(self, decision_type: DecisionType,
                             ic_surface_result: ICSurfaceResult,
                             stability_result: StabilityResult,
                             hierarchical_result: Optional[HierarchicalResult],
                             data: Optional[pd.DataFrame],
                             target: Optional[np.ndarray],
                             feature_name: str) -> LookbackSpec:
        """Create lookback specification based on decision type."""
        
        if decision_type == DecisionType.INACTIVE:
            return LookbackSpec(
                decision_type=DecisionType.INACTIVE,
                confidence_score=0.0,
                reasoning="Family marked as inactive due to poor performance"
            )
        
        elif decision_type == DecisionType.DISCRETE:
            # Use hierarchical result if available, otherwise use IC surface result
            if hierarchical_result:
                optimal_lookback = hierarchical_result.shrunk_lookbacks.get(
                    (ic_surface_result.family, ic_surface_result.family), 
                    ic_surface_result.optimal_lookback
                )
                confidence_score = 0.8
                reasoning = "Discrete choice using hierarchical shrinkage"
            else:
                optimal_lookback = ic_surface_result.optimal_lookback
                confidence_score = 0.7
                reasoning = "Discrete choice using IC surface optimization"
            
            # Snap to allowed discrete values
            optimal_lookback = self._snap_to_discrete_values(
                optimal_lookback, ic_surface_result.family
            )
            
            return LookbackSpec(
                decision_type=DecisionType.DISCRETE,
                primary_lookback=optimal_lookback,
                effective_lookback=optimal_lookback,
                confidence_score=confidence_score,
                reasoning=reasoning
            )
        
        elif decision_type == DecisionType.BLEND:
            # Select two nearby lookbacks for blending
            lookback1, lookback2 = self._select_blend_lookbacks(ic_surface_result)
            
            # Optimize blend weights if data is available
            if data is not None and target is not None and feature_name:
                w1, w2 = self.blend_optimizer.optimize_blend_weights(
                    data, target, ic_surface_result.family, feature_name, lookback1, lookback2
                )
            else:
                w1, w2 = 0.5, 0.5  # Equal weights as fallback
            
            effective_lookback = w1 * lookback1 + w2 * lookback2
            
            return LookbackSpec(
                decision_type=DecisionType.BLEND,
                primary_lookback=lookback1,
                secondary_lookback=lookback2,
                blend_weights=(w1, w2),
                effective_lookback=effective_lookback,
                confidence_score=0.6,
                reasoning="Blend approach for robustness"
            )
        
        else:
            # Default case
            return LookbackSpec(
                decision_type=DecisionType.DEFAULT,
                primary_lookback=self._get_default_lookback(ic_surface_result.family),
                effective_lookback=self._get_default_lookback(ic_surface_result.family),
                confidence_score=0.3,
                reasoning="Default fallback"
            )
    
    def _apply_hysteresis(self, symbol: str, family: FamilyType, 
                         lookback_spec: LookbackSpec, previous_lookback: float,
                         ic_surface_result: ICSurfaceResult) -> bool:
        """Apply hysteresis rules to determine if lookback should change."""
        if lookback_spec.effective_lookback is None:
            return False
        
        # Calculate expected IC gain
        expected_ic_gain = self._calculate_expected_ic_gain(
            ic_surface_result, lookback_spec, previous_lookback
        )
        
        # Apply hysteresis manager
        return self.hysteresis_manager.should_change_lookback(
            symbol, family, lookback_spec.effective_lookback, expected_ic_gain
        )
    
    def _calculate_expected_ic_gain(self, ic_surface_result: ICSurfaceResult,
                                   lookback_spec: LookbackSpec,
                                   previous_lookback: Optional[float]) -> float:
        """Calculate expected IC gain from lookback change."""
        if previous_lookback is None or lookback_spec.effective_lookback is None:
            return 0.0
        
        # Find closest lookback in IC surface
        lookbacks = ic_surface_result.lookbacks
        ic_values = ic_surface_result.ic_values
        
        # Interpolate IC for new lookback
        new_ic = np.interp(lookback_spec.effective_lookback, lookbacks, ic_values)
        
        # Interpolate IC for previous lookback
        prev_ic = np.interp(previous_lookback, lookbacks, ic_values)
        
        return new_ic - prev_ic
    
    def _snap_to_discrete_values(self, lookback: float, family: FamilyType) -> float:
        """Snap lookback to allowed discrete values."""
        allowed_windows = self.config.export.allowed_windows.get(family, [])
        if not allowed_windows:
            return lookback
        
        # Find closest allowed value
        distances = [abs(lookback - w) for w in allowed_windows]
        closest_idx = np.argmin(distances)
        return allowed_windows[closest_idx]
    
    def _select_blend_lookbacks(self, ic_surface_result: ICSurfaceResult) -> Tuple[float, float]:
        """Select two lookbacks for blending."""
        lookbacks = ic_surface_result.lookbacks
        ic_values = ic_surface_result.ic_values
        
        # Find the two highest IC values
        sorted_indices = np.argsort(ic_values)[::-1]
        
        if len(sorted_indices) >= 2:
            idx1, idx2 = sorted_indices[0], sorted_indices[1]
            return lookbacks[idx1], lookbacks[idx2]
        else:
            # Fallback: use optimal and a nearby value
            optimal_idx = np.argmax(ic_values)
            optimal_lookback = lookbacks[optimal_idx]
            
            # Find a nearby lookback
            if optimal_idx > 0:
                nearby_lookback = lookbacks[optimal_idx - 1]
            elif optimal_idx < len(lookbacks) - 1:
                nearby_lookback = lookbacks[optimal_idx + 1]
            else:
                nearby_lookback = optimal_lookback * 0.8  # 20% smaller
            
            return optimal_lookback, nearby_lookback
    
    def _get_default_lookback(self, family: FamilyType) -> float:
        """Get default lookback for a family."""
        default_lookbacks = {
            FamilyType.MOMENTUM: 12,
            FamilyType.VOLATILITY: 12,
            FamilyType.GK: 12,
            FamilyType.VWAP_ROLL: 12,
            FamilyType.RSI: 14,
            FamilyType.AUTOCORR: 12
        }
        return default_lookbacks.get(family, 12)


class MultiFamilyDecisionMaker:
    """Make decisions for multiple families and symbols."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.decision_maker = LookbackDecisionMaker(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def make_all_decisions(self, 
                          ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                          stability_results: Dict[str, Dict[FamilyType, StabilityResult]],
                          hierarchical_results: Dict[str, HierarchicalResult],
                          data: Dict[str, pd.DataFrame],
                          targets: Dict[str, np.ndarray],
                          feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, DecisionResult]]:
        """Make lookback decisions for all symbol-family combinations."""
        all_decisions = {}
        
        for symbol in ic_surface_results.keys():
            symbol_decisions = {}
            symbol_ic_results = ic_surface_results[symbol]
            symbol_stability_results = stability_results.get(symbol, {})
            symbol_hierarchical_result = hierarchical_results.get(symbol)
            symbol_data = data.get(symbol)
            symbol_target = targets.get(symbol)
            
            for family, ic_result in symbol_ic_results.items():
                try:
                    stability_result = symbol_stability_results.get(family)
                    if stability_result is None:
                        # Create dummy stability result
                        from .wf_stability import StabilityResult
                        stability_result = StabilityResult(
                            family=family,
                            global_optimal_lookback=ic_result.optimal_lookback,
                            global_optimal_ic=ic_result.optimal_ic,
                            fold_results=[],
                            match_rate=0.5,
                            average_ic_penalty=0.0,
                            average_lookback_difference=0.0,
                            stability_score=0.5,
                            recommendation="stable"
                        )
                    
                    decision = self.decision_maker.make_decision(
                        symbol=symbol,
                        family=family,
                        ic_surface_result=ic_result,
                        stability_result=stability_result,
                        hierarchical_result=symbol_hierarchical_result,
                        data=symbol_data,
                        target=symbol_target,
                        feature_name=feature_names.get(family, f"{family.value}_feature")
                    )
                    
                    symbol_decisions[family] = decision
                    
                except Exception as e:
                    self.logger.error(f"Failed to make decision for {symbol}-{family.value}: {e}")
                    continue
            
            all_decisions[symbol] = symbol_decisions
        
        return all_decisions
    
    def generate_decision_report(self, decisions: Dict[str, Dict[FamilyType, DecisionResult]]) -> Dict[str, Any]:
        """Generate comprehensive decision report."""
        report = {
            'summary': {
                'total_decisions': 0,
                'discrete_decisions': 0,
                'blend_decisions': 0,
                'default_decisions': 0,
                'inactive_decisions': 0,
                'average_confidence': 0.0,
                'families_with_changes': 0
            },
            'family_summary': {},
            'symbol_summary': {},
            'recommendations': []
        }
        
        all_confidence_scores = []
        families_with_changes = set()
        
        # Count by decision type
        decision_counts = {dt.value: 0 for dt in DecisionType}
        
        for symbol, symbol_decisions in decisions.items():
            symbol_changes = 0
            symbol_confidence = []
            
            for family, decision in symbol_decisions.items():
                report['summary']['total_decisions'] += 1
                decision_counts[decision.lookback_spec.decision_type.value] += 1
                
                all_confidence_scores.append(decision.lookback_spec.confidence_score)
                symbol_confidence.append(decision.lookback_spec.confidence_score)
                
                if decision.change_magnitude > 0.1:  # Significant change
                    symbol_changes += 1
                    families_with_changes.add(family)
                
                # Update family summary
                if family.value not in report['family_summary']:
                    report['family_summary'][family.value] = {
                        'total_decisions': 0,
                        'discrete_decisions': 0,
                        'blend_decisions': 0,
                        'default_decisions': 0,
                        'inactive_decisions': 0,
                        'average_confidence': 0.0
                    }
                
                family_summary = report['family_summary'][family.value]
                family_summary['total_decisions'] += 1
                family_summary[f"{decision.lookback_spec.decision_type.value}_decisions"] += 1
            
            # Update symbol summary
            report['symbol_summary'][symbol] = {
                'total_decisions': len(symbol_decisions),
                'changes': symbol_changes,
                'average_confidence': np.mean(symbol_confidence) if symbol_confidence else 0.0
            }
        
        # Update summary
        report['summary'].update(decision_counts)
        report['summary']['average_confidence'] = np.mean(all_confidence_scores) if all_confidence_scores else 0.0
        report['summary']['families_with_changes'] = len(families_with_changes)
        
        # Update family summaries with averages
        for family_summary in report['family_summary'].values():
            if family_summary['total_decisions'] > 0:
                family_summary['average_confidence'] = (
                    family_summary['average_confidence'] / family_summary['total_decisions']
                )
        
        # Generate recommendations
        if report['summary']['inactive_decisions'] > 0:
            report['recommendations'].append(
                f"Consider removing {report['summary']['inactive_decisions']} inactive families"
            )
        
        if report['summary']['blend_decisions'] > report['summary']['discrete_decisions']:
            report['recommendations'].append(
                "High number of blend decisions - consider reviewing stability thresholds"
            )
        
        if report['summary']['average_confidence'] < 0.6:
            report['recommendations'].append(
                "Low average confidence - consider increasing data quality or model complexity"
            )
        
        return report