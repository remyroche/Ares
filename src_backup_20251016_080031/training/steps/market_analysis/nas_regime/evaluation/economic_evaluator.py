"""
Economic Significance Evaluator for NAS Regime Detection.

This module provides economic significance evaluation for regime detection results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EconomicMetrics:
    """Economic significance metrics."""
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    return_ratio: float
    economic_significance: float

class EconomicSignificanceEvaluator:
    """Evaluates economic significance of regime detection results."""
    
    def __init__(self, significance_threshold: float = 0.05):
        """Initialize the economic significance evaluator."""
        self.significance_threshold = significance_threshold
        self.logger = logging.getLogger(__name__)
    
    def evaluate_regime_economic_significance(
        self, 
        regime_predictions: np.ndarray,
        market_data: pd.DataFrame,
        returns: np.ndarray
    ) -> EconomicMetrics:
        """Evaluate economic significance of regime predictions."""
        try:
            # Calculate basic metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            volatility = np.std(returns)
            return_ratio = np.mean(returns) / volatility if volatility > 0 else 0
            
            # Calculate economic significance
            economic_significance = self._calculate_economic_significance(
                sharpe_ratio, max_drawdown, volatility
            )
            
            return EconomicMetrics(
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                return_ratio=return_ratio,
                economic_significance=economic_significance
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating economic significance: {e}")
            return EconomicMetrics(0, 0, 0, 0, 0)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_economic_significance(
        self, 
        sharpe_ratio: float, 
        max_drawdown: float, 
        volatility: float
    ) -> float:
        """Calculate overall economic significance score."""
        # Normalize metrics
        sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1)  # Cap at 1
        drawdown_score = min(max(-max_drawdown / 0.2, 0), 1)  # Cap at 1
        volatility_score = min(max(1 - volatility / 0.3, 0), 1)  # Cap at 1
        
        # Weighted combination
        economic_significance = (
            0.4 * sharpe_score + 
            0.3 * drawdown_score + 
            0.3 * volatility_score
        )
        
        return economic_significance
