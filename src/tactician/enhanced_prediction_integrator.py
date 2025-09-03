# src/tactician/enhanced_prediction_integrator.py

"""
Thin integrator to unify enhanced scenario probabilities with multi-output target probabilities.
Provides helpers to map scenario analyses into price target hit probabilities.
"""

from __future__ import annotations

from typing import Any, Dict

def map_scenario_to_target_probabilities(scenario_analysis: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert scenario-level probabilities into approximate target-hit probabilities.

    Inputs:
      - scenario_analysis: dict with keys including 'profit_zone_probability' and 'risk_zone_probability'.

    Returns a dict of probabilities for 0.25%, 0.5%, 0.75%, 1.0% targets.
    """
    try:
        profit_p = float(scenario_analysis.get('profit_zone_probability', 0.0))
        # Distribute profit probability across tiers with decaying weights
        # 0.5% gets the most weight; longer targets receive less
        weights = {
            '0.25%': 0.35,
            '0.5%': 0.30,
            '0.75%': 0.20,
            '1.0%': 0.15,
        }
        total = sum(weights.values()) or 1.0
        return {k: min(1.0, max(0.0, profit_p * (w / total))) for k, w in weights.items()}
    except Exception:
        return {'0.25%': 0.30, '0.5%': 0.22, '0.75%': 0.15, '1.0%': 0.10}
