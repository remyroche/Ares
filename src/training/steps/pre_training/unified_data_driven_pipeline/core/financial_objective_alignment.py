"""
Financial Objective Alignment Framework

This module ensures that feature selection methods are properly aligned with
financial objectives and trading goals, optimizing for the most relevant
metrics for quantitative trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class FinancialObjective(Enum):
    """Financial objectives for feature selection alignment."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    TURNOVER = "turnover"
    STABILITY = "stability"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    EXPECTED_RETURN = "expected_return"
    VOLATILITY = "volatility"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"


class TradingRegime(Enum):
    """Trading regime types for context-aware feature selection."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    HIGH_FREQUENCY = "high_frequency"
    LOW_FREQUENCY = "low_frequency"


@dataclass
class FinancialObjectiveConfig:
    """Configuration for financial objective alignment."""
    primary_objectives: List[FinancialObjective]
    secondary_objectives: List[FinancialObjective]
    trading_regime: Optional[TradingRegime] = None
    risk_tolerance: float = 0.5  # 0-1 scale
    time_horizon: str = "medium_term"  # short_term, medium_term, long_term
    market_conditions: str = "normal"  # normal, volatile, crisis, recovery


@dataclass
class MethodAlignmentScore:
    """Alignment score for a feature selection method with financial objectives."""
    method_name: str
    overall_score: float
    objective_scores: Dict[FinancialObjective, float]
    regime_alignment: float
    risk_alignment: float
    time_horizon_alignment: float
    market_condition_alignment: float


class FinancialObjectiveAligner:
    """Aligns feature selection methods with financial objectives."""
    
    def __init__(self, config: Optional[FinancialObjectiveConfig] = None):
        """Initialize financial objective aligner."""
        self.config = config or FinancialObjectiveConfig(
            primary_objectives=[FinancialObjective.SHARPE_RATIO, FinancialObjective.MAX_DRAWDOWN],
            secondary_objectives=[FinancialObjective.STABILITY, FinancialObjective.TURNOVER]
        )
        self.logger = logger.getChild('FinancialObjectiveAligner')
        
        # Define method alignment profiles
        self.method_alignments = self._initialize_method_alignments()
        
        tprint_info("💰 Financial Objective Aligner initialized")
    
    def _initialize_method_alignments(self) -> Dict[str, Dict[FinancialObjective, float]]:
        """Initialize alignment scores for different feature selection methods."""
        return {
            # Standard methods
            'correlation': {
                FinancialObjective.SHARPE_RATIO: 0.6,
                FinancialObjective.MAX_DRAWDOWN: 0.5,
                FinancialObjective.CALMAR_RATIO: 0.6,
                FinancialObjective.SORTINO_RATIO: 0.5,
                FinancialObjective.INFORMATION_RATIO: 0.7,
                FinancialObjective.TURNOVER: 0.4,
                FinancialObjective.STABILITY: 0.8,
                FinancialObjective.PROFIT_FACTOR: 0.5,
                FinancialObjective.WIN_RATE: 0.6,
                FinancialObjective.EXPECTED_RETURN: 0.7,
                FinancialObjective.VOLATILITY: 0.8,
                FinancialObjective.SKEWNESS: 0.3,
                FinancialObjective.KURTOSIS: 0.3
            },
            'mutual_info': {
                FinancialObjective.SHARPE_RATIO: 0.7,
                FinancialObjective.MAX_DRAWDOWN: 0.6,
                FinancialObjective.CALMAR_RATIO: 0.7,
                FinancialObjective.SORTINO_RATIO: 0.6,
                FinancialObjective.INFORMATION_RATIO: 0.8,
                FinancialObjective.TURNOVER: 0.5,
                FinancialObjective.STABILITY: 0.7,
                FinancialObjective.PROFIT_FACTOR: 0.6,
                FinancialObjective.WIN_RATE: 0.7,
                FinancialObjective.EXPECTED_RETURN: 0.8,
                FinancialObjective.VOLATILITY: 0.7,
                FinancialObjective.SKEWNESS: 0.5,
                FinancialObjective.KURTOSIS: 0.5
            },
            'rfe': {
                FinancialObjective.SHARPE_RATIO: 0.8,
                FinancialObjective.MAX_DRAWDOWN: 0.7,
                FinancialObjective.CALMAR_RATIO: 0.8,
                FinancialObjective.SORTINO_RATIO: 0.7,
                FinancialObjective.INFORMATION_RATIO: 0.8,
                FinancialObjective.TURNOVER: 0.6,
                FinancialObjective.STABILITY: 0.9,
                FinancialObjective.PROFIT_FACTOR: 0.8,
                FinancialObjective.WIN_RATE: 0.8,
                FinancialObjective.EXPECTED_RETURN: 0.8,
                FinancialObjective.VOLATILITY: 0.8,
                FinancialObjective.SKEWNESS: 0.6,
                FinancialObjective.KURTOSIS: 0.6
            },
            'lasso': {
                FinancialObjective.SHARPE_RATIO: 0.8,
                FinancialObjective.MAX_DRAWDOWN: 0.8,
                FinancialObjective.CALMAR_RATIO: 0.8,
                FinancialObjective.SORTINO_RATIO: 0.8,
                FinancialObjective.INFORMATION_RATIO: 0.8,
                FinancialObjective.TURNOVER: 0.7,
                FinancialObjective.STABILITY: 0.8,
                FinancialObjective.PROFIT_FACTOR: 0.8,
                FinancialObjective.WIN_RATE: 0.8,
                FinancialObjective.EXPECTED_RETURN: 0.8,
                FinancialObjective.VOLATILITY: 0.8,
                FinancialObjective.SKEWNESS: 0.7,
                FinancialObjective.KURTOSIS: 0.7
            },
            
            # Enhanced methods
            'improved_mrmr': {
                FinancialObjective.SHARPE_RATIO: 0.9,
                FinancialObjective.MAX_DRAWDOWN: 0.8,
                FinancialObjective.CALMAR_RATIO: 0.9,
                FinancialObjective.SORTINO_RATIO: 0.8,
                FinancialObjective.INFORMATION_RATIO: 0.9,
                FinancialObjective.TURNOVER: 0.7,
                FinancialObjective.STABILITY: 0.9,
                FinancialObjective.PROFIT_FACTOR: 0.9,
                FinancialObjective.WIN_RATE: 0.8,
                FinancialObjective.EXPECTED_RETURN: 0.9,
                FinancialObjective.VOLATILITY: 0.8,
                FinancialObjective.SKEWNESS: 0.8,
                FinancialObjective.KURTOSIS: 0.8
            },
            'vectorbt_mrmr': {
                FinancialObjective.SHARPE_RATIO: 0.9,
                FinancialObjective.MAX_DRAWDOWN: 0.8,
                FinancialObjective.CALMAR_RATIO: 0.9,
                FinancialObjective.SORTINO_RATIO: 0.8,
                FinancialObjective.INFORMATION_RATIO: 0.9,
                FinancialObjective.TURNOVER: 0.7,
                FinancialObjective.STABILITY: 0.9,
                FinancialObjective.PROFIT_FACTOR: 0.9,
                FinancialObjective.WIN_RATE: 0.8,
                FinancialObjective.EXPECTED_RETURN: 0.9,
                FinancialObjective.VOLATILITY: 0.8,
                FinancialObjective.SKEWNESS: 0.8,
                FinancialObjective.KURTOSIS: 0.8
            },
            'vectorbt_rfe': {
                FinancialObjective.SHARPE_RATIO: 0.85,
                FinancialObjective.MAX_DRAWDOWN: 0.8,
                FinancialObjective.CALMAR_RATIO: 0.85,
                FinancialObjective.SORTINO_RATIO: 0.8,
                FinancialObjective.INFORMATION_RATIO: 0.85,
                FinancialObjective.TURNOVER: 0.7,
                FinancialObjective.STABILITY: 0.9,
                FinancialObjective.PROFIT_FACTOR: 0.85,
                FinancialObjective.WIN_RATE: 0.8,
                FinancialObjective.EXPECTED_RETURN: 0.85,
                FinancialObjective.VOLATILITY: 0.8,
                FinancialObjective.SKEWNESS: 0.7,
                FinancialObjective.KURTOSIS: 0.7
            },
            'vectorbt_lasso': {
                FinancialObjective.SHARPE_RATIO: 0.85,
                FinancialObjective.MAX_DRAWDOWN: 0.85,
                FinancialObjective.CALMAR_RATIO: 0.85,
                FinancialObjective.SORTINO_RATIO: 0.85,
                FinancialObjective.INFORMATION_RATIO: 0.85,
                FinancialObjective.TURNOVER: 0.8,
                FinancialObjective.STABILITY: 0.85,
                FinancialObjective.PROFIT_FACTOR: 0.85,
                FinancialObjective.WIN_RATE: 0.85,
                FinancialObjective.EXPECTED_RETURN: 0.85,
                FinancialObjective.VOLATILITY: 0.85,
                FinancialObjective.SKEWNESS: 0.8,
                FinancialObjective.KURTOSIS: 0.8
            },
            'enhanced_ensemble': {
                FinancialObjective.SHARPE_RATIO: 0.95,
                FinancialObjective.MAX_DRAWDOWN: 0.9,
                FinancialObjective.CALMAR_RATIO: 0.95,
                FinancialObjective.SORTINO_RATIO: 0.9,
                FinancialObjective.INFORMATION_RATIO: 0.95,
                FinancialObjective.TURNOVER: 0.8,
                FinancialObjective.STABILITY: 0.95,
                FinancialObjective.PROFIT_FACTOR: 0.95,
                FinancialObjective.WIN_RATE: 0.9,
                FinancialObjective.EXPECTED_RETURN: 0.95,
                FinancialObjective.VOLATILITY: 0.9,
                FinancialObjective.SKEWNESS: 0.9,
                FinancialObjective.KURTOSIS: 0.9
            },
            'enhanced_advanced': {
                FinancialObjective.SHARPE_RATIO: 0.95,
                FinancialObjective.MAX_DRAWDOWN: 0.9,
                FinancialObjective.CALMAR_RATIO: 0.95,
                FinancialObjective.SORTINO_RATIO: 0.9,
                FinancialObjective.INFORMATION_RATIO: 0.95,
                FinancialObjective.TURNOVER: 0.8,
                FinancialObjective.STABILITY: 0.95,
                FinancialObjective.PROFIT_FACTOR: 0.95,
                FinancialObjective.WIN_RATE: 0.9,
                FinancialObjective.EXPECTED_RETURN: 0.95,
                FinancialObjective.VOLATILITY: 0.9,
                FinancialObjective.SKEWNESS: 0.9,
                FinancialObjective.KURTOSIS: 0.9
            }
        }
    
    def calculate_method_alignment(self, method_name: str) -> MethodAlignmentScore:
        """Calculate alignment score for a specific method."""
        if method_name not in self.method_alignments:
            tprint_warning(f"Unknown method: {method_name}")
            return MethodAlignmentScore(
                method_name=method_name,
                overall_score=0.0,
                objective_scores={},
                regime_alignment=0.0,
                risk_alignment=0.0,
                time_horizon_alignment=0.0,
                market_condition_alignment=0.0
            )
        
        method_scores = self.method_alignments[method_name]
        
        # Calculate objective alignment
        primary_scores = [method_scores.get(obj, 0.0) for obj in self.config.primary_objectives]
        secondary_scores = [method_scores.get(obj, 0.0) for obj in self.config.secondary_objectives]
        
        # Weighted objective score
        objective_score = (
            np.mean(primary_scores) * 0.7 +  # 70% weight for primary objectives
            np.mean(secondary_scores) * 0.3   # 30% weight for secondary objectives
        )
        
        # Calculate regime alignment
        regime_alignment = self._calculate_regime_alignment(method_name)
        
        # Calculate risk alignment
        risk_alignment = self._calculate_risk_alignment(method_name)
        
        # Calculate time horizon alignment
        time_horizon_alignment = self._calculate_time_horizon_alignment(method_name)
        
        # Calculate market condition alignment
        market_condition_alignment = self._calculate_market_condition_alignment(method_name)
        
        # Overall score
        overall_score = (
            objective_score * 0.4 +
            regime_alignment * 0.2 +
            risk_alignment * 0.2 +
            time_horizon_alignment * 0.1 +
            market_condition_alignment * 0.1
        )
        
        return MethodAlignmentScore(
            method_name=method_name,
            overall_score=overall_score,
            objective_scores=method_scores,
            regime_alignment=regime_alignment,
            risk_alignment=risk_alignment,
            time_horizon_alignment=time_horizon_alignment,
            market_condition_alignment=market_condition_alignment
        )
    
    def _calculate_regime_alignment(self, method_name: str) -> float:
        """Calculate alignment with current trading regime."""
        if not self.config.trading_regime:
            return 0.8  # Neutral if no regime specified
        
        regime_alignments = {
            TradingRegime.TRENDING: {
                'correlation': 0.9,
                'mutual_info': 0.8,
                'rfe': 0.8,
                'lasso': 0.8,
                'improved_mrmr': 0.9,
                'vectorbt_mrmr': 0.9,
                'vectorbt_rfe': 0.8,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.95,
                'enhanced_advanced': 0.95
            },
            TradingRegime.MEAN_REVERTING: {
                'correlation': 0.7,
                'mutual_info': 0.8,
                'rfe': 0.9,
                'lasso': 0.9,
                'improved_mrmr': 0.9,
                'vectorbt_mrmr': 0.9,
                'vectorbt_rfe': 0.9,
                'vectorbt_lasso': 0.9,
                'enhanced_ensemble': 0.95,
                'enhanced_advanced': 0.95
            },
            TradingRegime.VOLATILE: {
                'correlation': 0.6,
                'mutual_info': 0.7,
                'rfe': 0.8,
                'lasso': 0.8,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.8,
                'vectorbt_rfe': 0.8,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.9,
                'enhanced_advanced': 0.9
            },
            TradingRegime.LOW_VOLATILITY: {
                'correlation': 0.8,
                'mutual_info': 0.8,
                'rfe': 0.7,
                'lasso': 0.7,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.8,
                'vectorbt_rfe': 0.7,
                'vectorbt_lasso': 0.7,
                'enhanced_ensemble': 0.8,
                'enhanced_advanced': 0.8
            },
            TradingRegime.HIGH_FREQUENCY: {
                'correlation': 0.9,
                'mutual_info': 0.8,
                'rfe': 0.6,
                'lasso': 0.7,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.9,
                'vectorbt_rfe': 0.7,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.8,
                'enhanced_advanced': 0.8
            },
            TradingRegime.LOW_FREQUENCY: {
                'correlation': 0.6,
                'mutual_info': 0.8,
                'rfe': 0.9,
                'lasso': 0.9,
                'improved_mrmr': 0.9,
                'vectorbt_mrmr': 0.8,
                'vectorbt_rfe': 0.9,
                'vectorbt_lasso': 0.9,
                'enhanced_ensemble': 0.95,
                'enhanced_advanced': 0.95
            }
        }
        
        return regime_alignments.get(self.config.trading_regime, {}).get(method_name, 0.8)
    
    def _calculate_risk_alignment(self, method_name: str) -> float:
        """Calculate alignment with risk tolerance."""
        # Methods that are more conservative (higher stability, lower turnover)
        conservative_methods = ['rfe', 'lasso', 'enhanced_ensemble', 'enhanced_advanced']
        aggressive_methods = ['correlation', 'mutual_info', 'improved_mrmr', 'vectorbt_mrmr']
        
        if method_name in conservative_methods:
            # Conservative methods align well with low risk tolerance
            return 1.0 - abs(self.config.risk_tolerance - 0.3)
        elif method_name in aggressive_methods:
            # Aggressive methods align well with high risk tolerance
            return 1.0 - abs(self.config.risk_tolerance - 0.7)
        else:
            # Neutral methods
            return 0.8
    
    def _calculate_time_horizon_alignment(self, method_name: str) -> float:
        """Calculate alignment with time horizon."""
        time_horizon_alignments = {
            'short_term': {
                'correlation': 0.9,
                'mutual_info': 0.8,
                'rfe': 0.6,
                'lasso': 0.7,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.9,
                'vectorbt_rfe': 0.7,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.8,
                'enhanced_advanced': 0.8
            },
            'medium_term': {
                'correlation': 0.7,
                'mutual_info': 0.8,
                'rfe': 0.8,
                'lasso': 0.8,
                'improved_mrmr': 0.9,
                'vectorbt_mrmr': 0.8,
                'vectorbt_rfe': 0.8,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.9,
                'enhanced_advanced': 0.9
            },
            'long_term': {
                'correlation': 0.6,
                'mutual_info': 0.7,
                'rfe': 0.9,
                'lasso': 0.9,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.7,
                'vectorbt_rfe': 0.9,
                'vectorbt_lasso': 0.9,
                'enhanced_ensemble': 0.95,
                'enhanced_advanced': 0.95
            }
        }
        
        return time_horizon_alignments.get(self.config.time_horizon, {}).get(method_name, 0.8)
    
    def _calculate_market_condition_alignment(self, method_name: str) -> float:
        """Calculate alignment with market conditions."""
        market_condition_alignments = {
            'normal': {
                'correlation': 0.8,
                'mutual_info': 0.8,
                'rfe': 0.8,
                'lasso': 0.8,
                'improved_mrmr': 0.9,
                'vectorbt_mrmr': 0.9,
                'vectorbt_rfe': 0.8,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.9,
                'enhanced_advanced': 0.9
            },
            'volatile': {
                'correlation': 0.6,
                'mutual_info': 0.7,
                'rfe': 0.8,
                'lasso': 0.8,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.8,
                'vectorbt_rfe': 0.8,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.9,
                'enhanced_advanced': 0.9
            },
            'crisis': {
                'correlation': 0.5,
                'mutual_info': 0.6,
                'rfe': 0.7,
                'lasso': 0.7,
                'improved_mrmr': 0.7,
                'vectorbt_mrmr': 0.7,
                'vectorbt_rfe': 0.7,
                'vectorbt_lasso': 0.7,
                'enhanced_ensemble': 0.8,
                'enhanced_advanced': 0.8
            },
            'recovery': {
                'correlation': 0.7,
                'mutual_info': 0.8,
                'rfe': 0.8,
                'lasso': 0.8,
                'improved_mrmr': 0.8,
                'vectorbt_mrmr': 0.8,
                'vectorbt_rfe': 0.8,
                'vectorbt_lasso': 0.8,
                'enhanced_ensemble': 0.9,
                'enhanced_advanced': 0.9
            }
        }
        
        return market_condition_alignments.get(self.config.market_conditions, {}).get(method_name, 0.8)
    
    def rank_methods_by_alignment(self, available_methods: List[str]) -> List[Tuple[str, float]]:
        """Rank methods by their alignment with financial objectives."""
        method_scores = []
        
        for method_name in available_methods:
            alignment = self.calculate_method_alignment(method_name)
            method_scores.append((method_name, alignment.overall_score))
        
        # Sort by score (descending)
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        return method_scores
    
    def get_recommended_methods(self, available_methods: List[str], 
                              top_k: int = 3) -> List[str]:
        """Get top-k recommended methods based on financial objective alignment."""
        ranked_methods = self.rank_methods_by_alignment(available_methods)
        return [method for method, score in ranked_methods[:top_k]]
    
    def get_alignment_report(self, method_name: str) -> Dict[str, Any]:
        """Get detailed alignment report for a method."""
        alignment = self.calculate_method_alignment(method_name)
        
        return {
            'method_name': method_name,
            'overall_score': alignment.overall_score,
            'primary_objective_scores': {
                obj.value: alignment.objective_scores.get(obj, 0.0) 
                for obj in self.config.primary_objectives
            },
            'secondary_objective_scores': {
                obj.value: alignment.objective_scores.get(obj, 0.0) 
                for obj in self.config.secondary_objectives
            },
            'regime_alignment': alignment.regime_alignment,
            'risk_alignment': alignment.risk_alignment,
            'time_horizon_alignment': alignment.time_horizon_alignment,
            'market_condition_alignment': alignment.market_condition_alignment,
            'recommendation': self._get_recommendation(alignment.overall_score)
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on alignment score."""
        if score >= 0.9:
            return "Highly recommended"
        elif score >= 0.8:
            return "Recommended"
        elif score >= 0.7:
            return "Moderately recommended"
        elif score >= 0.6:
            return "Consider with caution"
        else:
            return "Not recommended"


# Convenience functions
def create_financial_objective_aligner(config: Optional[FinancialObjectiveConfig] = None) -> FinancialObjectiveAligner:
    """Create a financial objective aligner with optional configuration."""
    return FinancialObjectiveAligner(config)


def get_financially_aligned_methods(available_methods: List[str],
                                  primary_objectives: List[FinancialObjective],
                                  secondary_objectives: List[FinancialObjective],
                                  trading_regime: Optional[TradingRegime] = None,
                                  risk_tolerance: float = 0.5) -> List[str]:
    """Get financially aligned methods for given objectives."""
    config = FinancialObjectiveConfig(
        primary_objectives=primary_objectives,
        secondary_objectives=secondary_objectives,
        trading_regime=trading_regime,
        risk_tolerance=risk_tolerance
    )
    
    aligner = create_financial_objective_aligner(config)
    return aligner.get_recommended_methods(available_methods)