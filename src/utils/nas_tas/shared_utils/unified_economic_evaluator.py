"""Unified Economic Evaluator for NAS and TAS Systems"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

class EvaluationType(Enum):
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"

class ArchitectureType(Enum):
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"

@dataclass
class EconomicEvaluationConfig:
    evaluation_types: List[EvaluationType] = field(default_factory=lambda: [
        EvaluationType.ECONOMIC_SIGNIFICANCE,
        EvaluationType.TRADING_VIABILITY
    ])
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    significance_threshold: float = 0.05
    risk_free_rate: float = 0.02

@dataclass
class EconomicMetrics:
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    economic_significance: float
    trading_viability: float
    overall_score: float

@dataclass
class EconomicEvaluationResult:
    economic_metrics: EconomicMetrics
    evaluation_summary: Dict[str, Any]
    architecture_type: ArchitectureType
    total_evaluation_time: float
    success: bool = True
    error_message: Optional[str] = None

class UnifiedEconomicEvaluator:
    def __init__(self, config: Optional[EconomicEvaluationConfig] = None):
        self.config = config or EconomicEvaluationConfig()
        print(f"🚀 Unified Economic Evaluator initialized - {self.config.architecture_type.value}")
    
    def evaluate(self, predictions: np.ndarray, market_data: pd.DataFrame, 
                returns: np.ndarray, architecture_params: Optional[Dict[str, Any]] = None) -> EconomicEvaluationResult:
        try:
            print("💰 Starting economic evaluation...")
            start_time = time.time()
            
            self._validate_inputs(predictions, market_data, returns)
            economic_metrics = self._calculate_economic_metrics(returns, predictions)
            evaluation_summary = self._generate_evaluation_summary(economic_metrics)
            
            result = EconomicEvaluationResult(
                economic_metrics=economic_metrics,
                evaluation_summary=evaluation_summary,
                architecture_type=self.config.architecture_type,
                total_evaluation_time=time.time() - start_time
            )
            
            print(f"✅ Economic evaluation completed in {result.total_evaluation_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Economic evaluation failed: {e}")
            return EconomicEvaluationResult(
                economic_metrics=EconomicMetrics(0, 0, 0, 0, 0, 0),
                evaluation_summary={},
                architecture_type=self.config.architecture_type,
                total_evaluation_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _validate_inputs(self, predictions: np.ndarray, market_data: pd.DataFrame, returns: np.ndarray):
        if len(predictions) == 0 or len(returns) == 0:
            raise ValueError("Predictions and returns cannot be empty")
        if len(predictions) != len(returns):
            raise ValueError("Predictions and returns must have the same length")
    
    def _calculate_economic_metrics(self, returns: np.ndarray, predictions: np.ndarray) -> EconomicMetrics:
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        
        economic_significance = self._calculate_economic_significance(sharpe_ratio, max_drawdown, volatility)
        trading_viability = self._calculate_trading_viability(returns, predictions)
        overall_score = (economic_significance + trading_viability) / 2.0
        
        return EconomicMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            economic_significance=economic_significance,
            trading_viability=trading_viability,
            overall_score=overall_score
        )
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - self.config.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_economic_significance(self, sharpe_ratio: float, max_drawdown: float, volatility: float) -> float:
        sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1)
        drawdown_score = min(max(-max_drawdown / 0.2, 0), 1)
        volatility_score = min(max(1 - volatility / 0.3, 0), 1)
        return 0.4 * sharpe_score + 0.3 * drawdown_score + 0.3 * volatility_score
    
    def _calculate_trading_viability(self, returns: np.ndarray, predictions: np.ndarray) -> float:
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        win_rate_score = min(max(win_rate / 0.6, 0), 1)
        profit_factor_score = min(max(profit_factor / 2.0, 0), 1)
        
        return 0.6 * win_rate_score + 0.4 * profit_factor_score
    
    def _generate_evaluation_summary(self, economic_metrics: EconomicMetrics) -> Dict[str, Any]:
        return {
            'overall_assessment': 'excellent' if economic_metrics.overall_score > 0.8 else
                                'good' if economic_metrics.overall_score > 0.6 else
                                'fair' if economic_metrics.overall_score > 0.4 else 'poor',
            'economic_significance_level': 'high' if economic_metrics.economic_significance > 0.7 else
                                         'medium' if economic_metrics.economic_significance > 0.4 else 'low',
            'trading_viability_level': 'high' if economic_metrics.trading_viability > 0.7 else
                                     'medium' if economic_metrics.trading_viability > 0.4 else 'low'
        }

def create_unified_economic_evaluator(config: Optional[EconomicEvaluationConfig] = None) -> UnifiedEconomicEvaluator:
    return UnifiedEconomicEvaluator(config)

__all__ = [
    'UnifiedEconomicEvaluator',
    'EconomicEvaluationConfig',
    'EconomicEvaluationResult',
    'EconomicMetrics',
    'EvaluationType',
    'ArchitectureType',
    'create_unified_economic_evaluator'
]
