"""
Regime Utilities

Utility functions for hybrid regime detection and modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging


class RegimeUtils:
    """Utility functions for regime detection and modeling."""
    
    @staticmethod
    def calculate_regime_stability(regime_predictions: np.ndarray, window_size: int = 10) -> np.ndarray:
        """Calculate regime stability scores."""
        if len(regime_predictions) < 2:
            return np.array([1.0] * len(regime_predictions))
        
        stability_scores = np.zeros(len(regime_predictions))
        
        for i in range(len(regime_predictions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(regime_predictions), i + window_size // 2 + 1)
            
            window_regimes = regime_predictions[start_idx:end_idx]
            current_regime = regime_predictions[i]
            
            consistency = np.sum(window_regimes == current_regime) / len(window_regimes)
            stability_scores[i] = consistency
        
        return stability_scores
    
    @staticmethod
    def calculate_regime_transitions(regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime transition probabilities."""
        if len(regime_predictions) < 2:
            return np.array([0.0] * len(regime_predictions))
        
        transitions = np.zeros(len(regime_predictions))
        
        for i in range(1, len(regime_predictions)):
            transitions[i] = 1.0 if regime_predictions[i] != regime_predictions[i-1] else 0.0
        
        return transitions
    
    @staticmethod
    def generate_regime_labels(regime_predictions: np.ndarray) -> List[str]:
        """Generate regime labels from predictions."""
        unique_regimes = np.unique(regime_predictions)
        regime_labels = []
        
        for regime_id in unique_regimes:
            if regime_id == 0:
                regime_labels.append("normal")
            elif regime_id == 1:
                regime_labels.append("bull_market")
            elif regime_id == 2:
                regime_labels.append("bear_market")
            elif regime_id == 3:
                regime_labels.append("high_volatility")
            elif regime_id == 4:
                regime_labels.append("low_volatility")
            elif regime_id == 5:
                regime_labels.append("trending_up")
            elif regime_id == 6:
                regime_labels.append("trending_down")
            elif regime_id == 7:
                regime_labels.append("mean_reverting")
            elif regime_id == 8:
                regime_labels.append("breakout")
            elif regime_id == 9:
                regime_labels.append("consolidation")
            elif regime_id == 10:
                regime_labels.append("crisis")
            else:
                regime_labels.append("unknown")
        
        return regime_labels