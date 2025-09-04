# kelly_criterion_fix.py
"""
Kelly Criterion calculation utilities for position sizing.
"""

import numpy as np
from typing import Dict, Any


def calculate_correct_kelly_position_size(
    price_target_confidences: Dict[str, float],
    adversarial_confidences: Dict[str, float],
    kelly_multiplier: float = 0.25,
    min_position_size: float = 0.01,
    max_position_size: float = 0.5,
) -> float:
    """
    Calculate position size using Kelly Criterion based on ML confidence scores.
    
    Args:
        price_target_confidences: Dictionary of price target levels to confidence scores
        adversarial_confidences: Dictionary of adversarial movement levels to risk scores
        kelly_multiplier: Kelly fraction multiplier (fractional Kelly)
        min_position_size: Minimum position size
        max_position_size: Maximum position size
        
    Returns:
        float: Calculated position size
    """
    try:
        # Extract average win probability from price target confidences
        if price_target_confidences:
            win_probs = list(price_target_confidences.values())
            p = np.mean(win_probs) if win_probs else 0.5
        else:
            p = 0.5
            
        # Extract average loss probability from adversarial confidences
        if adversarial_confidences:
            loss_probs = list(adversarial_confidences.values())
            q = np.mean(loss_probs) if loss_probs else 0.5
        else:
            q = 0.5
            
        # Ensure probabilities are valid
        p = max(0.01, min(0.99, p))
        q = max(0.01, min(0.99, q))
        
        # Calculate win/loss ratio (b)
        # Assuming symmetric payoffs for simplicity
        b = 1.0  # 1:1 risk-reward ratio
        
        # Kelly formula: f = (p*b - q) / b
        # where f is the fraction of capital to risk
        kelly_fraction = (p * b - q) / b
        
        # Apply fractional Kelly
        position_size = kelly_fraction * kelly_multiplier
        
        # Ensure within bounds
        position_size = max(min_position_size, min(max_position_size, position_size))
        
        return position_size
        
    except Exception as e:
        # Return minimum position size on error
        return min_position_size