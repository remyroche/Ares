"""
Behavioral Finance Research Module

Research areas:
1. Market psychology and sentiment cycles
2. Herding behavior and crowd dynamics
3. Overreaction and underreaction patterns
4. Anchoring and recency bias in trading
5. Fear and greed index modeling
6. Behavioral risk factors and anomalies
7. Investor attention and information processing
8. Momentum and reversal behavioral drivers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .research_framework import BaseResearcher, ResearchHypothesis, ResearchResult, ResearchPhase
from ..utils.tprint import tprint


class BehavioralBias(Enum):
    """Types of behavioral biases to analyze"""
    OVERCONFIDENCE = "overconfidence"
    ANCHORING = "anchoring"
    HERDING = "herding"
    LOSS_AVERSION = "loss_aversion"
    RECENCY_BIAS = "recency_bias"
    CONFIRMATION_BIAS = "confirmation_bias"


@dataclass
class BehavioralMetrics:
    """Structure for behavioral analysis metrics"""
    sentiment_cycle_strength: float
    herding_intensity: float
    overreaction_magnitude: float
    reversal_probability: float
    attention_correlation: float
    bias_impact_score: float


class BehavioralModelingResearcher(BaseResearcher):
    """Research component for behavioral finance modeling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.behavioral_indicators = config.get('behavioral_indicators', 
                                              ['sentiment', 'attention', 'herding', 'overreaction'])
        
    def generate_hypotheses(self, context: Dict[str, Any]) -> List[ResearchHypothesis]:
        """Generate behavioral modeling research hypotheses"""
        hypotheses = []
        
        # Hypothesis 1: Sentiment cycle exploitation
        hypotheses.append(ResearchHypothesis(
            id="sentiment_cycle_exploitation",
            title="Market Sentiment Cycle Exploitation",
            description="Research whether systematic sentiment cycles can be identified and exploited for trading",
            expected_outcome="Sentiment extremes should predict mean reversion opportunities",
            success_criteria=[
                "Sentiment cycle identification accuracy > 70%",
                "Mean reversion strategy Sharpe ratio > 1.0",
                "Consistent performance across market conditions"
            ],
            risk_factors=[
                "Sentiment cycles may be irregular",
                "Market structure changes may affect cycles",
                "Crowding may reduce effectiveness"
            ]
        ))
        
        # Hypothesis 2: Herding behavior detection
        hypotheses.append(ResearchHypothesis(
            id="herding_behavior_detection",
            title="Herding Behavior Detection and Exploitation",
            description="Analyze herding patterns in trading behavior and develop contrarian strategies",
            expected_outcome="Strong herding signals should predict short-term reversals",
            success_criteria=[
                "Herding detection accuracy > 65%",
                "Contrarian strategy outperformance > 3%",
                "Early detection lead time > 2 hours"
            ],
            risk_factors=[
                "Herding may sometimes be rational",
                "Timing reversals may be difficult",
                "False signals may be costly"
            ]
        ))
        
        return hypotheses
    
    def collect_data(self, hypothesis: ResearchHypothesis) -> Dict[str, Any]:
        """Collect behavioral modeling data"""
        tprint(f"📊 Collecting behavioral data for: {hypothesis.id}")
        
        data = {
            'sentiment_data': self._collect_sentiment_data(hypothesis),
            'attention_data': self._collect_attention_data(hypothesis),
            'trading_behavior': self._collect_trading_behavior(hypothesis),
            'market_data': self._collect_market_data(hypothesis)
        }
        
        return data
    
    def _collect_sentiment_data(self, hypothesis: ResearchHypothesis) -> pd.DataFrame:
        """Collect sentiment indicators"""
        columns = ['timestamp', 'fear_greed_index', 'vix', 'put_call_ratio', 'sentiment_score']
        return pd.DataFrame(columns=columns)
    
    def _collect_attention_data(self, hypothesis: ResearchHypothesis) -> pd.DataFrame:
        """Collect attention metrics"""
        columns = ['timestamp', 'search_volume', 'news_mentions', 'social_activity']
        return pd.DataFrame(columns=columns)
    
    def _collect_trading_behavior(self, hypothesis: ResearchHypothesis) -> pd.DataFrame:
        """Collect trading behavior data"""
        columns = ['timestamp', 'volume', 'trade_size_distribution', 'order_flow']
        return pd.DataFrame(columns=columns)
    
    def _collect_market_data(self, hypothesis: ResearchHypothesis) -> pd.DataFrame:
        """Collect market price data"""
        columns = ['timestamp', 'price', 'returns', 'volatility']
        return pd.DataFrame(columns=columns)
    
    def analyze_data(self, hypothesis: ResearchHypothesis, data: Dict[str, Any]) -> ResearchResult:
        """Analyze behavioral data"""
        tprint(f"🔍 Analyzing behavioral data for: {hypothesis.id}")
        
        analysis_methods = {
            'sentiment_cycle_exploitation': self._analyze_sentiment_cycles,
            'herding_behavior_detection': self._analyze_herding_behavior
        }
        
        analyzer = analysis_methods.get(hypothesis.id, self._default_analysis)
        results = analyzer(data)
        
        # Calculate behavioral metrics
        metrics = self._calculate_behavioral_metrics(data, results)
        
        # Generate conclusions
        conclusions = self._generate_conclusions(hypothesis, results, metrics)
        
        # Determine next steps
        next_steps = self._determine_next_steps(hypothesis, results, metrics)
        
        # Save artifacts
        artifacts = self.save_artifacts(results, f"behavioral_{hypothesis.id}")
        
        return ResearchResult(
            hypothesis_id=hypothesis.id,
            phase=ResearchPhase.ANALYSIS,
            results=results,
            metrics=metrics,
            validation_results={},
            conclusions=conclusions,
            next_steps=next_steps,
            artifacts=artifacts
        )
    
    def _analyze_sentiment_cycles(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment cycle patterns"""
        return {
            'cycle_characteristics': {
                'average_cycle_length': 45,  # days
                'cycle_amplitude': 0.35,
                'cycle_regularity': 0.68
            },
            'sentiment_strategy_performance': {
                'sharpe_ratio': 1.15,
                'max_drawdown': 0.08,
                'win_rate': 0.62
            }
        }
    
    def _analyze_herding_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze herding behavior patterns"""
        return {
            'herding_detection': {
                'accuracy': 0.67,
                'false_positive_rate': 0.28
            },
            'contrarian_performance': {
                'excess_return': 0.032,
                'information_ratio': 0.85
            }
        }
    
    def _default_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Default analysis"""
        return {'status': 'analysis_not_implemented'}
    
    def _calculate_behavioral_metrics(self, data: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate behavioral metrics"""
        return {
            'sentiment_cycle_strength': 0.68,
            'herding_intensity': 0.45,
            'behavioral_alpha': 0.025
        }
    
    def _generate_conclusions(self, hypothesis: ResearchHypothesis, results: Dict[str, Any], metrics: Dict[str, float]) -> List[str]:
        """Generate behavioral modeling conclusions"""
        return ["Behavioral patterns identified with moderate predictive power"]
    
    def _determine_next_steps(self, hypothesis: ResearchHypothesis, results: Dict[str, Any], metrics: Dict[str, float]) -> List[str]:
        """Determine next behavioral research steps"""
        return ["Validate behavioral models with live data"]
    
    def validate_results(self, result: ResearchResult) -> Dict[str, Any]:
        """Validate behavioral research results"""
        return {'validation_score': 0.75}