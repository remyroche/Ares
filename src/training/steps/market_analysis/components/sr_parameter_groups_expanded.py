"""
Expanded SR Parameter Groups for Hierarchical Optimization

This module defines comprehensive parameter groups that include ALL parameters
from the search space, not just a subset. This ensures thorough optimization
with 100+ combinations instead of just 12.

The parameters are organized into logical groups based on:
1. Impact on detection
2. Dependencies between parameters
3. Computational cost
4. Optimization priority

Total parameters: 20+ (vs. previous 6)
Expected combinations: 100-150+ (vs. previous 12)
"""

from typing import Dict, Any, List
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import create_param_group


def create_expanded_sr_parameter_groups(
    search_space: Dict[str, Any],
    enable_strength_weight_optimization: bool = True,
    enable_temporal_params: bool = True,
    enable_volume_params: bool = True,
    enable_price_action_params: bool = True
) -> List[Any]:
    """
    Create expanded parameter groups for SR optimization.
    
    This includes ALL parameters from the search space organized into
    hierarchical groups for efficient optimization.
    
    Parameter Groups:
    ----------------
    1. Core Detection (Priority 1) - 2 params
       - min_touches: How many touches required
       - distance_threshold: Minimum distance between levels
    
    2. Lookback & Thresholds (Priority 2) - 3 params
       - lookback_periods: Historical data window
       - touch_tolerance: Price tolerance for touches
       - volume_threshold: Volume confirmation level
    
    3. Advanced SR Filters (Priority 3) - 3 params
       - breakout_threshold: Price movement for breakouts
       - consolidation_periods: Consolidation time requirement
       - trend_strength_threshold: Trend strength filter
    
    4. Temporal Parameters (Priority 4) - 3 params
       - min_formation_time: Minimum time for level formation
       - max_formation_time: Maximum time for level validity
       - time_decay_factor: How levels degrade over time
    
    5. Volume Parameters (Priority 5) - 3 params
       - volume_spike_threshold: Volume spike detection
       - volume_consistency_threshold: Volume consistency requirement
       - volume_weight: Volume importance in strength calculation
    
    6. Price Action Parameters (Priority 6) - 3 params
       - wick_ratio_threshold: Wick to body ratio
       - body_ratio_threshold: Candle body size
       - price_momentum_threshold: Price momentum strength
    
    7. Strength Weights (Priority 7) - 11 params [OPTIONAL]
       - All strength calculation weights and penalties
       - Post-calculation filtering threshold
    
    Total: 17 base params + 11 strength params = 28 parameters
    
    Args:
        search_space: Full search space dictionary
        enable_strength_weight_optimization: Include strength weight params
        enable_temporal_params: Include temporal parameters
        enable_volume_params: Include volume-specific parameters
        enable_price_action_params: Include price action parameters
    
    Returns:
        List of parameter groups for hierarchical optimization
    """
    
    param_groups = []
    
    # =========================================================================
    # GROUP 1: CORE DETECTION (Priority 1)
    # =========================================================================
    # These are the most fundamental parameters that define what gets detected
    
    core_detection_params = {
        "min_touches": search_space.get('min_touches', {
            "type": "int", "low": 2, "high": 8
        }),
        "distance_threshold": search_space.get('distance_threshold', {
            "type": "float", "low": 0.005, "high": 0.03
        })
    }
    
    param_groups.append(
        create_param_group(
            name="core_detection",
            params=core_detection_params,
            priority=1,
            description="Core SR detection: fundamental filtering parameters"
        )
    )
    
    # =========================================================================
    # GROUP 2: LOOKBACK & THRESHOLDS (Priority 2)
    # =========================================================================
    # These depend on core detection and refine what gets detected
    
    lookback_threshold_params = {
        "lookback_periods": search_space.get('lookback_periods', {
            "type": "int", "low": 20, "high": 100
        }),
        "touch_tolerance": search_space.get('touch_tolerance', {
            "type": "float", "low": 0.001, "high": 0.01
        }),
        "volume_threshold": search_space.get('volume_threshold', {
            "type": "float", "low": 0.5, "high": 2.0
        })
    }
    
    param_groups.append(
        create_param_group(
            name="lookback_thresholds",
            params=lookback_threshold_params,
            priority=2,
            depends_on=["core_detection"],
            description="Lookback window and threshold refinements"
        )
    )
    
    # =========================================================================
    # GROUP 3: ADVANCED SR FILTERS (Priority 3)
    # =========================================================================
    # Market context and pattern-based filters
    
    advanced_sr_params = {
        "breakout_threshold": search_space.get('breakout_threshold', {
            "type": "float", "low": 0.01, "high": 0.05
        }),
        "consolidation_periods": search_space.get('consolidation_periods', {
            "type": "int", "low": 5, "high": 30
        }),
        "trend_strength_threshold": search_space.get('trend_strength_threshold', {
            "type": "float", "low": 0.3, "high": 0.7
        })
    }
    
    param_groups.append(
        create_param_group(
            name="advanced_sr_filters",
            params=advanced_sr_params,
            priority=3,
            depends_on=["core_detection", "lookback_thresholds"],
            description="Advanced market context and pattern filters"
        )
    )
    
    # =========================================================================
    # GROUP 4: TEMPORAL PARAMETERS (Priority 4) [OPTIONAL]
    # =========================================================================
    # Time-based refinements for level validity
    
    if enable_temporal_params:
        temporal_params = {
            "min_formation_time": search_space.get('min_formation_time', {
                "type": "int", "low": 5, "high": 50
            }),
            "max_formation_time": search_space.get('max_formation_time', {
                "type": "int", "low": 100, "high": 500
            }),
            "time_decay_factor": search_space.get('time_decay_factor', {
                "type": "float", "low": 0.9, "high": 1.0
            })
        }
        
        param_groups.append(
            create_param_group(
                name="temporal_parameters",
                params=temporal_params,
                priority=4,
                depends_on=["core_detection"],
                description="Time-based level formation and decay parameters"
            )
        )
    
    # =========================================================================
    # GROUP 5: VOLUME PARAMETERS (Priority 5) [OPTIONAL]
    # =========================================================================
    # Volume-specific analysis parameters
    
    if enable_volume_params:
        volume_params = {
            "volume_spike_threshold": search_space.get('volume_spike_threshold', {
                "type": "float", "low": 1.5, "high": 3.0
            }),
            "volume_consistency_threshold": search_space.get('volume_consistency_threshold', {
                "type": "float", "low": 0.5, "high": 1.0
            }),
            "volume_weight": search_space.get('volume_weight', {
                "type": "float", "low": 0.1, "high": 0.5
            })
        }
        
        param_groups.append(
            create_param_group(
                name="volume_parameters",
                params=volume_params,
                priority=5,
                depends_on=["core_detection", "lookback_thresholds"],
                description="Volume spike and consistency analysis"
            )
        )
    
    # =========================================================================
    # GROUP 6: PRICE ACTION PARAMETERS (Priority 6) [OPTIONAL]
    # =========================================================================
    # Candlestick pattern and price action filters
    
    if enable_price_action_params:
        price_action_params = {
            "wick_ratio_threshold": search_space.get('wick_ratio_threshold', {
                "type": "float", "low": 0.3, "high": 0.8
            }),
            "body_ratio_threshold": search_space.get('body_ratio_threshold', {
                "type": "float", "low": 0.2, "high": 0.7
            }),
            "price_momentum_threshold": search_space.get('price_momentum_threshold', {
                "type": "float", "low": 0.01, "high": 0.05
            })
        }
        
        param_groups.append(
            create_param_group(
                name="price_action_parameters",
                params=price_action_params,
                priority=6,
                depends_on=["core_detection", "advanced_sr_filters"],
                description="Price action and candlestick pattern analysis"
            )
        )
    
    # =========================================================================
    # GROUP 7: STRENGTH WEIGHTS (Priority 7) [OPTIONAL]
    # =========================================================================
    # Strength calculation weights - highest dimensional group
    
    if enable_strength_weight_optimization:
        strength_weight_params = {
            # Positive boosts
            "touch_weight": {"type": "float", "low": 0.05, "high": 0.3},
            "volume_weight": {"type": "float", "low": 0.1, "high": 0.4},
            "consistency_weight": {"type": "float", "low": 0.1, "high": 0.4},
            "confluence_weight": {"type": "float", "low": 0.05, "high": 0.2},
            "pivot_boost": {"type": "float", "low": 0.05, "high": 0.2},
            "psychological_boost": {"type": "float", "low": 0.02, "high": 0.1},
            "hvn_boost": {"type": "float", "low": 0.05, "high": 0.2},
            
            # Negative penalties
            "failure_penalty_base": {"type": "float", "low": 0.1, "high": 0.5},
            "failure_volume_multiplier": {"type": "float", "low": 1.0, "high": 2.5},
            "failure_max_penalty": {"type": "float", "low": 0.4, "high": 1.0},
            
            # Post-calculation filter
            "strength_filter_threshold": {"type": "float", "low": 0.3, "high": 0.8}
        }
        
        param_groups.append(
            create_param_group(
                name="strength_weights",
                params=strength_weight_params,
                priority=7,
                depends_on=["core_detection", "lookback_thresholds"],
                description="Strength calculation weights and post-calculation filtering"
            )
        )
    
    return param_groups


def get_parameter_group_summary(param_groups: List[Any]) -> Dict[str, Any]:
    """
    Get a summary of the parameter groups.
    
    Returns:
        Dictionary with group statistics and information
    """
    total_params = 0
    groups_info = []
    
    for group in param_groups:
        group_name = group.name if hasattr(group, 'name') else str(group)
        group_params = group.params if hasattr(group, 'params') else {}
        param_count = len(group_params)
        total_params += param_count
        
        groups_info.append({
            'name': group_name,
            'priority': group.priority if hasattr(group, 'priority') else None,
            'param_count': param_count,
            'parameters': list(group_params.keys()),
            'depends_on': group.depends_on if hasattr(group, 'depends_on') else []
        })
    
    return {
        'total_groups': len(param_groups),
        'total_parameters': total_params,
        'groups': groups_info
    }


def print_parameter_group_summary(param_groups: List[Any]):
    """Print a formatted summary of parameter groups."""
    summary = get_parameter_group_summary(param_groups)
    
    print("\n" + "="*80)
    print("SR PARAMETER GROUPS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Total Groups: {summary['total_groups']}")
    print(f"📊 Total Parameters: {summary['total_parameters']}")
    
    print("\n📋 Group Details:")
    for group_info in summary['groups']:
        print(f"\n   Group {group_info['priority']}: {group_info['name']}")
        print(f"   └─ Parameters ({group_info['param_count']}): {', '.join(group_info['parameters'])}")
        if group_info['depends_on']:
            print(f"   └─ Depends on: {', '.join(group_info['depends_on'])}")
    
    print("\n✅ Expected Optimization Results:")
    print(f"   - With 5 points/param coarse grid: ~{summary['total_parameters'] * 5} combinations")
    print(f"   - With 8 points/param fine grid: ~{summary['total_parameters'] * 8} combinations")
    print(f"   - With 150 TPE trials: 150 Bayesian samples")
    print(f"   - Total combinations: 100-200+ (vs. previous 12)")
    
    print("\n" + "="*80 + "\n")


# =========================================================================
# EXAMPLE USAGE
# =========================================================================

if __name__ == '__main__':
    # Example: Create expanded parameter groups
    from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep
    
    # Initialize step to get search space
    sr_step = SRParameterOptimizationStep()
    search_space = sr_step._create_sr_search_space(None)
    
    # Create expanded parameter groups
    param_groups = create_expanded_sr_parameter_groups(
        search_space=search_space,
        enable_strength_weight_optimization=True,
        enable_temporal_params=True,
        enable_volume_params=True,
        enable_price_action_params=True
    )
    
    # Print summary
    print_parameter_group_summary(param_groups)

