"""
Multi-Horizon Decision Logic - Updated for Multi-Horizon Profit Labeling

This module provides decision logic for trading based on multi-horizon profit
probability predictions, replacing binary classification decisions with
probability-based trading strategies.

Key features:
- Probability-based trading decisions
- High-leverage optimized strategies
- Multi-horizon opportunity assessment
- Risk-aware position sizing
- Continuous reassessment logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

# Optimized imports using common utilities
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    safe_divide, safe_mean, safe_std, safe_percentage_change,
    validate_finite, validate_positive, validate_range,
    timed_operation, memory_checkpoint, gpu_context
)
from src.utils.math_validation import (
    safe_correlation, safe_covariance, safe_weighted_average,
    safe_kelly_calculation, math_safe
)
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

@dataclass
class TradingDecisionConfig:
    """Configuration for multi-horizon trading decisions."""
    
    # High-leverage strategy parameters
    leverage_multiplier: float = 50.0  # High leverage trading
    max_account_risk_per_trade: float = 0.02  # 2% max account risk
    
    # Probability thresholds for different actions (SHORT-TERM FOCUSED)
    entry_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'micro_immediate_prob': 0.75,     # 75% for micro moves (10 min)
        'small_immediate_prob': 0.70,     # 70% for small moves (10 min)
        'medium_immediate_prob': 0.65,    # 65% for medium moves (10 min)
        'good_immediate_prob': 0.60,      # 60% for good moves (10 min)
        'micro_short_prob': 0.70,         # 70% for micro moves (20 min)
        'small_short_prob': 0.65,         # 65% for small moves (20 min)
        'medium_short_prob': 0.60,        # 60% for medium moves (20 min)
        'good_short_prob': 0.55,          # 55% for good moves (20 min)
        'overall_opportunity': 0.60,      # 60% overall opportunity (lower for more trades)
        'leverage_adjusted_score': 0.65,  # 65% leverage-adjusted score
        'reversal_capture_score': 0.60    # 60% reversal capture score
    })
    
    # Exit thresholds
    exit_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'profit_protection': 0.50,        # Protect 50% of unrealized gains
        'opportunity_decline': 0.40,      # Exit if opportunity drops below 40%
        'time_based_exit': 0.30,          # Exit if probability drops to 30%
        'stop_loss_trigger': 0.20         # Hard stop if opportunity drops to 20%
    })
    
    # Position sizing parameters
    position_sizing: Dict[str, float] = field(default_factory=lambda: {
        'base_size': 0.01,                # 1% base position size
        'confidence_multiplier': 2.0,     # Multiply by confidence
        'leverage_efficiency': 0.8,       # Use 80% of available leverage
        'max_position_size': 0.05,        # 5% maximum position size
        'min_position_size': 0.002        # 0.2% minimum position size
    })
    
    # Time-based parameters (SHORT-TERM FOCUSED) - configurable
    immediate_timeframe_minutes: int = 15    # Immediate opportunity timeframe (configurable)
    short_term_timeframe_minutes: int = 30   # Short-term opportunity timeframe (configurable)
    max_position_duration: int = 25          # 25 minutes maximum hold time
    reassessment_frequency: int = 3          # Reassess every 3 minutes (more frequent)
    
    # Risk management
    transaction_cost: float = 0.001      # 0.1% transaction cost
    
    # Strategy weights for different scenarios (SHORT-TERM FOCUSED)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'immediate_scalp': 0.6,           # Quick scalping (10 min) - Higher weight
        'short_term_swing': 0.4,          # Short swing (20 min) - Lower weight
        'reversal_capture': 0.3           # NEW: Reversal capture strategy
    })

@dataclass
class TradingDecision:
    """Result of trading decision analysis."""
    action: str                          # 'buy', 'sell', 'hold', 'exit'
    confidence: float                    # Confidence in decision [0,1]
    position_size: float                 # Recommended position size
    expected_profit: float               # Expected profit percentage
    max_risk: float                      # Maximum risk percentage
    time_horizon: int                    # Expected time horizon (minutes)
    reasoning: List[str]                 # Decision reasoning
    probabilities: Dict[str, float]      # Key probabilities used
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiHorizonDecisionEngine:
    """
    Decision engine for multi-horizon profit probability trading.
    
    Makes trading decisions based on probability distributions rather than
    binary signals, optimized for high-leverage trading strategies.
    """
    
    def __init__(self, config: Optional[TradingDecisionConfig] = None):
        """Initialize the decision engine with hardware optimizations."""
        self.config = config or TradingDecisionConfig()
        self.logger = get_logger('MultiHorizonDecisionEngine')
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Optimize CPU for mathematical operations
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
        
        # Track active positions for reassessment
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f'🎯 Multi-Horizon Decision Engine initialized with M1 optimizations')
        self.logger.info(f'   → Leverage: {self.config.leverage_multiplier}x')
        self.logger.info(f'   → Max account risk: {self.config.max_account_risk_per_trade*100:.1f}%')
        self.logger.info(f'   → Entry threshold (overall): {self.config.entry_thresholds["overall_opportunity"]*100:.1f}%')
    
    @timed_operation
    @traced(span_name='make_trading_decision')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=TradingDecision('hold', 0.0, 0.0, 0.0, 0.0, 0, [], {}))
    @log_execution_time()
    def make_trading_decision(self, 
                            predictions: Dict[str, float],
                            current_price: float,
                            market_data: Optional[pd.DataFrame] = None,
                            position_id: Optional[str] = None) -> TradingDecision:
        """
        Make trading decision based on multi-horizon probability predictions.
        
        Args:
            predictions: Dictionary of probability predictions from model
            current_price: Current market price
            market_data: Recent market data for context
            position_id: ID of existing position (if any)
            
        Returns:
            TradingDecision with recommended action and parameters
        """
        self.logger.debug(f'🤔 Making trading decision for price {current_price}')
        
        # Use memory checkpoint for decision making
        with memory_checkpoint('trading_decision'):
            # Check if this is for an existing position
            if position_id and position_id in self.active_positions:
                return self._reassess_existing_position(position_id, predictions, current_price)
            
            # Analyze entry opportunities with GPU acceleration if available
            with gpu_context('entry_analysis') if self.gpu_manager else memory_checkpoint('entry_analysis'):
                entry_analysis = self._analyze_entry_opportunities(predictions, current_price)
        
            if entry_analysis['should_enter']:
                decision = self._create_entry_decision(entry_analysis, predictions, current_price)
                
                # Track position if entered
                if decision.action in ['buy', 'sell']:
                    self._track_new_position(position_id or f'pos_{int(datetime.now().timestamp())}', 
                                           decision, predictions, current_price)
            else:
                decision = TradingDecision(
                    action='hold',
                    confidence=entry_analysis['confidence'],
                    position_size=0.0,
                    expected_profit=0.0,
                    max_risk=0.0,
                    time_horizon=0,
                    reasoning=entry_analysis['reasoning'],
                    probabilities=self._extract_key_probabilities(predictions)
                )
        
        self.logger.debug(f'📊 Decision: {decision.action} (confidence: {decision.confidence:.3f})')
        return decision
    
    def _analyze_entry_opportunities(self, predictions: Dict[str, float], current_price: float) -> Dict[str, Any]:
        """Analyze entry opportunities across all time horizons."""
        analysis = {
            'should_enter': False,
            'confidence': 0.0,
            'best_strategy': None,
            'reasoning': [],
            'opportunities': {}
        }
        
        # Analyze immediate opportunities (configurable timeframe)
        immediate_opp = self._analyze_immediate_opportunities(predictions)
        analysis['opportunities']['immediate'] = immediate_opp
        
        # Analyze short-term opportunities (configurable timeframe)
        short_term_opp = self._analyze_short_term_opportunities(predictions)
        analysis['opportunities']['short_term'] = short_term_opp
        
        # Analyze reversal capture opportunities (NEW)
        reversal_opp = self._analyze_reversal_opportunities(predictions)
        analysis['opportunities']['reversal_capture'] = reversal_opp
        
        # Determine best overall opportunity
        best_opportunity = self._select_best_opportunity(analysis['opportunities'])
        
        if best_opportunity['score'] > 0.6:  # Minimum threshold for entry
            analysis['should_enter'] = True
            analysis['confidence'] = best_opportunity['score']
            analysis['best_strategy'] = best_opportunity['strategy']
            analysis['reasoning'] = best_opportunity['reasoning']
        else:
            analysis['reasoning'] = ['No high-confidence opportunities found']
        
        return analysis
    
    def _analyze_immediate_opportunities(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze immediate opportunities (configurable timeframe, default ~15 minutes) for micro price movements."""
        immediate_probs = {
            'micro': validate_finite(predictions.get('micro_immediate_prob', 0.0), 'micro_immediate_prob'),
            'small': validate_finite(predictions.get('small_immediate_prob', 0.0), 'small_immediate_prob'),
            'medium': validate_finite(predictions.get('medium_immediate_prob', 0.0), 'medium_immediate_prob')
        }
        
        # High-leverage strategy: prioritize high-probability small moves using safe operations
        weighted_score = safe_weighted_average(
            [immediate_probs['micro'], immediate_probs['small'], immediate_probs['medium']],
            [0.5, 0.3, 0.2]  # weights: micro=50%, small=30%, medium=20%
        )
        
        reasoning = []
        if immediate_probs['micro'] > 0.8:
            reasoning.append(f"High micro move probability: {immediate_probs['micro']:.1%}")
        if immediate_probs['small'] > 0.75:
            reasoning.append(f"High small move probability: {immediate_probs['small']:.1%}")
        if weighted_score > 0.7:
            reasoning.append(f"Strong immediate opportunity score: {weighted_score:.1%}")
        
        return {
            'score': weighted_score,
            'strategy': 'immediate_scalp',
            'time_horizon': self.config.immediate_timeframe_minutes,
            'probabilities': immediate_probs,
            'reasoning': reasoning
        }
    
    def _analyze_short_term_opportunities(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze short-term opportunities (30 minutes) with safe math operations."""
        short_term_probs = {
            'micro': validate_finite(predictions.get('micro_short_prob', 0.0), 'micro_short_prob'),
            'small': validate_finite(predictions.get('small_short_prob', 0.0), 'small_short_prob'),
            'medium': validate_finite(predictions.get('medium_short_prob', 0.0), 'medium_short_prob'),
            'good': validate_finite(predictions.get('good_short_prob', 0.0), 'good_short_prob')
        }
        
        # Balanced approach for short-term using safe operations
        weighted_score = safe_weighted_average(
            [short_term_probs['micro'], short_term_probs['small'], short_term_probs['medium'], short_term_probs['good']],
            [0.3, 0.4, 0.2, 0.1]  # weights: micro=30%, small=40%, medium=20%, good=10%
        )
        
        reasoning = []
        if short_term_probs['small'] > 0.7:
            reasoning.append(f"High short-term small move probability: {short_term_probs['small']:.1%}")
        if short_term_probs['medium'] > 0.65:
            reasoning.append(f"Good medium move probability: {short_term_probs['medium']:.1%}")
        if weighted_score > 0.65:
            reasoning.append(f"Strong short-term opportunity: {weighted_score:.1%}")
        
        return {
            'score': weighted_score,
            'strategy': 'short_term_swing',
            'time_horizon': self.config.short_term_timeframe_minutes,
            'probabilities': short_term_probs,
            'reasoning': reasoning
        }
    
    def _analyze_reversal_opportunities(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze reversal capture opportunities for close/reopen strategies.
        
        This focuses on identifying small reversals and corrections that allow
        for profitable close/reopen positions around minor market movements.
        """
        reversal_probs = {
            'reversal_capture_score': validate_finite(predictions.get('reversal_capture_score', 0.0), 'reversal_capture_score'),
            'immediate_micro': validate_finite(predictions.get('micro_immediate_prob', 0.0), 'immediate_micro'),
            'immediate_small': validate_finite(predictions.get('small_immediate_prob', 0.0), 'immediate_small'),
            'reassessment_freq': validate_finite(predictions.get('reassessment_frequency', 5.0), 'reassessment_freq')
        }
        
        # Calculate reversal opportunity score with safe operations
        reversal_score = reversal_probs['reversal_capture_score']
        immediate_strength = safe_mean(pd.Series([reversal_probs['immediate_micro'], reversal_probs['immediate_small']]), default=0.0)
        
        # Weight reversal capture with immediate strength using safe operations
        weighted_score = safe_weighted_average(
            [reversal_score, immediate_strength],
            [0.6, 0.4]  # weights: reversal=60%, immediate=40%
        )
        
        reasoning = []
        if reversal_score > 0.6:
            reasoning.append(f"High reversal capture score: {reversal_score:.1%}")
        if immediate_strength > 0.7:
            reasoning.append(f"Strong immediate opportunities: {immediate_strength:.1%}")
        if reversal_probs['reassessment_freq'] <= 3.0:
            reasoning.append(f"High-frequency reassessment optimal: {reversal_probs['reassessment_freq']:.1f}min")
        if weighted_score > 0.6:
            reasoning.append(f"Good reversal capture opportunity: {weighted_score:.1%}")
        
        return {
            'score': weighted_score,
            'strategy': 'reversal_capture',
            'time_horizon': int(reversal_probs['reassessment_freq']),  # Dynamic based on reassessment
            'probabilities': reversal_probs,
            'reasoning': reasoning
        }
    
    def _select_best_opportunity(self, opportunities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best opportunity across all time horizons."""
        best_opp = {'score': 0.0, 'strategy': None, 'reasoning': []}
        
        # Apply strategy weights
        for opp_name, opp_data in opportunities.items():
            strategy = opp_data['strategy']
            weight = self.config.strategy_weights.get(strategy, 1.0)
            weighted_score = opp_data['score'] * weight
            
            if weighted_score > best_opp['score']:
                best_opp = {
                    'score': weighted_score,
                    'strategy': strategy,
                    'time_horizon': opp_data['time_horizon'],
                    'reasoning': opp_data['reasoning'],
                    'probabilities': opp_data['probabilities']
                }
        
        return best_opp
    
    def _create_entry_decision(self, analysis: Dict[str, Any], 
                             predictions: Dict[str, float], 
                             current_price: float) -> TradingDecision:
        """Create entry decision based on analysis."""
        strategy = analysis['best_strategy']
        confidence = analysis['confidence']
        
        # Calculate position size based on confidence and strategy
        position_size = self._calculate_position_size(confidence, strategy)
        
        # Estimate expected profit and risk
        expected_profit = self._estimate_expected_profit(strategy, predictions)
        max_risk = self._calculate_max_risk(position_size)
        
        # Determine direction (assume bullish for now - can be enhanced)
        action = 'buy'  # Can be enhanced with directional analysis
        
        return TradingDecision(
            action=action,
            confidence=confidence,
            position_size=position_size,
            expected_profit=expected_profit,
            max_risk=max_risk,
            time_horizon=analysis['opportunities'][strategy.replace('_', '_').split('_')[0]]['time_horizon'],
            reasoning=analysis['reasoning'],
            probabilities=self._extract_key_probabilities(predictions),
            metadata={
                'strategy': strategy,
                'entry_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _calculate_position_size(self, confidence: float, strategy: str) -> float:
        """Calculate position size based on confidence and strategy with safe operations."""
        base_size = validate_finite(self.config.position_sizing['base_size'], 'base_size')
        confidence_multiplier = validate_finite(self.config.position_sizing['confidence_multiplier'], 'confidence_multiplier')
        confidence = validate_finite(confidence, 'confidence')
        
        # Adjust base size for strategy
        strategy_multipliers = {
            'immediate_scalp': 1.2,    # Slightly larger for quick scalps
            'short_term_swing': 1.0,   # Standard size
            'medium_term_hold': 0.8,   # Smaller for longer holds
            'reversal_capture': 0.9    # Moderate size for reversal trades
        }
        
        strategy_multiplier = validate_finite(strategy_multipliers.get(strategy, 1.0), 'strategy_multiplier')
        
        # Calculate size using safe operations
        raw_size = base_size * confidence * confidence_multiplier * strategy_multiplier
        raw_size = validate_finite(raw_size, 'raw_position_size')
        
        # Apply limits
        min_size = validate_finite(self.config.position_sizing['min_position_size'], 'min_position_size')
        max_size = validate_finite(self.config.position_sizing['max_position_size'], 'max_position_size')
        
        position_size = np.clip(raw_size, min_size, max_size)
        
        return validate_finite(position_size, 'final_position_size')
    
    def _estimate_expected_profit(self, strategy: str, predictions: Dict[str, float]) -> float:
        """Estimate expected profit based on strategy and predictions with safe operations."""
        # Simplified profit estimation - can be enhanced
        profit_estimates = {
            'immediate_scalp': 0.003,    # 0.3% average for scalping
            'short_term_swing': 0.005,   # 0.5% average for short swing
            'medium_term_hold': 0.008    # 0.8% average for medium hold
        }
        
        base_profit = validate_finite(profit_estimates.get(strategy, 0.005), 'base_profit')
        
        # Adjust based on overall opportunity score
        overall_opp = validate_finite(predictions.get('overall_opportunity', 0.5), 'overall_opportunity')
        adjusted_profit = base_profit * overall_opp
        adjusted_profit = validate_finite(adjusted_profit, 'adjusted_profit')
        
        # Subtract transaction costs
        transaction_cost = validate_finite(self.config.transaction_cost, 'transaction_cost')
        net_profit = adjusted_profit - transaction_cost
        
        return validate_finite(max(0.0, net_profit), 'net_profit')
    
    def _calculate_max_risk(self, position_size: float) -> float:
        """Calculate maximum risk based on position size and leverage with safe operations."""
        # Account risk = position_size * leverage * price_risk
        # For high leverage, we use tight stops
        price_risk_pct = validate_finite(0.002, 'price_risk_pct')  # 0.2% typical stop loss
        position_size = validate_finite(position_size, 'position_size')
        leverage = validate_finite(self.config.leverage_multiplier, 'leverage')
        
        account_risk = position_size * leverage * price_risk_pct
        account_risk = validate_finite(account_risk, 'account_risk')
        
        # Cap at max account risk
        max_account_risk = validate_finite(self.config.max_account_risk_per_trade, 'max_account_risk')
        
        return validate_finite(min(account_risk, max_account_risk), 'final_max_risk')
    
    def _extract_key_probabilities(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Extract key probabilities for decision record."""
        key_probs = {}
        
        important_keys = [
            'micro_immediate_prob', 'small_immediate_prob', 'medium_immediate_prob',
            'small_short_prob', 'medium_short_prob',
            'overall_opportunity', 'leverage_adjusted_score'
        ]
        
        for key in important_keys:
            if key in predictions:
                key_probs[key] = predictions[key]
        
        return key_probs
    
    def _calculate_confidence_score(self, predictions: Dict[str, float]) -> float:
        """Calculate simple confidence score based on predictions."""
        try:
            # Simple confidence based on overall opportunity
            overall_opportunity = predictions.get('overall_opportunity', 0.5)
            leverage_adjusted = predictions.get('leverage_adjusted_score', 0.5)
            
            # Average the main prediction scores
            confidence = (overall_opportunity + leverage_adjusted) / 2
            
            return np.clip(confidence, 0.1, 0.95)  # Keep within reasonable bounds
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate confidence score: {e}")
            return 0.5  # Default moderate confidence
    
    def _track_new_position(self, position_id: str, decision: TradingDecision, 
                          predictions: Dict[str, float], entry_price: float):
        """Track new position for reassessment."""
        self.active_positions[position_id] = {
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'entry_decision': decision,
            'entry_predictions': predictions,
            'last_reassessment': datetime.now(),
            'reassessment_count': 0
        }
        
        self.logger.info(f'📈 Tracking new position {position_id}: {decision.action} at {entry_price}')
    
    def _reassess_existing_position(self, position_id: str, 
                                  current_predictions: Dict[str, float], 
                                  current_price: float) -> TradingDecision:
        """Reassess existing position based on current conditions."""
        position = self.active_positions[position_id]
        
        # Calculate current P&L
        entry_price = position['entry_price']
        entry_decision = position['entry_decision']
        
        if entry_decision.action == 'buy':
            current_pnl = (current_price - entry_price) / entry_price
        else:  # sell
            current_pnl = (entry_price - current_price) / entry_price
        
        # Check exit conditions
        exit_analysis = self._analyze_exit_conditions(
            position, current_predictions, current_pnl
        )
        
        if exit_analysis['should_exit']:
            # Remove from active positions
            del self.active_positions[position_id]
            
            return TradingDecision(
                action='exit',
                confidence=exit_analysis['confidence'],
                position_size=0.0,  # Exit full position
                expected_profit=current_pnl,
                max_risk=0.0,
                time_horizon=0,
                reasoning=exit_analysis['reasoning'],
                probabilities=self._extract_key_probabilities(current_predictions),
                metadata={
                    'exit_type': exit_analysis['exit_type'],
                    'current_pnl': current_pnl,
                    'position_duration': (datetime.now() - position['entry_time']).seconds // 60
                }
            )
        else:
            # Update reassessment info
            position['last_reassessment'] = datetime.now()
            position['reassessment_count'] += 1
            
            return TradingDecision(
                action='hold',
                confidence=0.5,
                position_size=entry_decision.position_size,
                expected_profit=current_pnl,
                max_risk=entry_decision.max_risk,
                time_horizon=entry_decision.time_horizon,
                reasoning=['Position maintained - conditions still favorable'],
                probabilities=self._extract_key_probabilities(current_predictions),
                metadata={'current_pnl': current_pnl}
            )
    
    def _analyze_exit_conditions(self, position: Dict[str, Any], 
                               current_predictions: Dict[str, float], 
                               current_pnl: float) -> Dict[str, Any]:
        """Analyze whether to exit existing position."""
        exit_analysis = {
            'should_exit': False,
            'confidence': 0.0,
            'exit_type': None,
            'reasoning': []
        }
        
        # Time-based exit
        position_duration = (datetime.now() - position['entry_time']).seconds // 60
        if position_duration >= self.config.max_position_duration:
            exit_analysis.update({
                'should_exit': True,
                'confidence': 0.8,
                'exit_type': 'time_based',
                'reasoning': [f'Maximum position duration reached: {position_duration} minutes']
            })
            return exit_analysis
        
        # Profit protection
        if current_pnl > 0.002:  # If profitable
            protection_threshold = current_pnl * self.config.exit_thresholds['profit_protection']
            if current_pnl < protection_threshold:
                exit_analysis.update({
                    'should_exit': True,
                    'confidence': 0.7,
                    'exit_type': 'profit_protection',
                    'reasoning': [f'Profit protection triggered: {current_pnl:.1%} < {protection_threshold:.1%}']
                })
                return exit_analysis
        
        # Opportunity decline
        current_opportunity = current_predictions.get('overall_opportunity', 0.0)
        if current_opportunity < self.config.exit_thresholds['opportunity_decline']:
            exit_analysis.update({
                'should_exit': True,
                'confidence': 0.6,
                'exit_type': 'opportunity_decline',
                'reasoning': [f'Opportunity declined: {current_opportunity:.1%}']
            })
            return exit_analysis
        
        # Stop loss
        if current_pnl < -0.002:  # If losing more than 0.2%
            exit_analysis.update({
                'should_exit': True,
                'confidence': 0.9,
                'exit_type': 'stop_loss',
                'reasoning': [f'Stop loss triggered: {current_pnl:.1%}']
            })
            return exit_analysis
        
        return exit_analysis

# Convenience functions
def create_decision_engine(config: Optional[TradingDecisionConfig] = None) -> MultiHorizonDecisionEngine:
    """Create multi-horizon decision engine."""
    return MultiHorizonDecisionEngine(config)

def make_trading_decision(predictions: Dict[str, float], 
                        current_price: float,
                        config: Optional[TradingDecisionConfig] = None) -> TradingDecision:
    """Make trading decision based on predictions."""
    engine = MultiHorizonDecisionEngine(config)
    return engine.make_trading_decision(predictions, current_price)

# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Multi-Horizon Decision Logic')
    
    # Test predictions (simulated model output)
    test_predictions = {
        # Immediate probabilities
        'micro_immediate_prob': 0.85,
        'small_immediate_prob': 0.75,
        'medium_immediate_prob': 0.45,
        
        # Short-term probabilities
        'micro_short_prob': 0.90,
        'small_short_prob': 0.80,
        'medium_short_prob': 0.60,
        'good_short_prob': 0.35,
        
        # Medium-term probabilities
        'small_medium_prob': 0.85,
        'medium_medium_prob': 0.70,
        'good_medium_prob': 0.50,
        'great_medium_prob': 0.25,
        
        # Composite scores
        'immediate_opportunity': 0.78,
        'short_term_opportunity': 0.71,
        'medium_term_opportunity': 0.65,
        'overall_opportunity': 0.72,
        'leverage_adjusted_score': 0.76
    }
    
    # Test decision making
    config = TradingDecisionConfig()
    engine = MultiHorizonDecisionEngine(config)
    
    decision = engine.make_trading_decision(
        predictions=test_predictions,
        current_price=100.0
    )
    
    tprint(f'📊 Trading Decision:')
    tprint(f'   → Action: {decision.action}')
    tprint(f'   → Confidence: {decision.confidence:.1%}')
    tprint(f'   → Position size: {decision.position_size:.2%}')
    tprint(f'   → Expected profit: {decision.expected_profit:.2%}')
    tprint(f'   → Max risk: {decision.max_risk:.2%}')
    tprint(f'   → Time horizon: {decision.time_horizon} minutes')
    
    tprint(f'\n💭 Reasoning:')
    for reason in decision.reasoning:
        tprint(f'   → {reason}')
    
    tprint('✅ Multi-Horizon Decision Logic test completed!')