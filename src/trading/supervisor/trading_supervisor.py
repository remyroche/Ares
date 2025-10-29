"""
Trading Supervisor

Meta-coordinator and Risk Oversight component for trading operations.

Responsibilities:
- Portfolio-level risk oversight (aggregate across all positions)
- Cross-asset position sizing review (avoid over-correlation)
- Circuit breakers and fail-safes
- System health monitoring
- Execution quality monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..config.trading_config import TradingConfig
from ..utils.error_handling import (
    TradingError, TradingErrorSeverity, trading_error_handler
)
from ..sizing.risk_calculator import RiskCalculator

logger = system_logger.getChild('TradingSupervisor')


class SupervisorStatus(Enum):
    """Supervisor operational status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ValidationResult:
    """Pre-decision validation result."""
    is_valid: bool
    reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionApproval:
    """Decision validation result."""
    approved: bool
    confidence_modifier: float = 1.0  # Multiplier for confidence adjustment
    reason: str = ""
    risk_adjustments: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionCheck:
    """Pre-execution check result."""
    can_proceed: bool
    reason: str = ""
    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    trigger_reason: str = ""
    cooldown_until: Optional[datetime] = None
    trigger_count: int = 0


class TradingSupervisor:
    """
    Trading Supervisor - Meta-coordinator and Risk Oversight
    
    Provides:
    - Portfolio-level risk oversight
    - Cross-asset position sizing review (prevents over-correlation)
    - Circuit breakers and fail-safes
    - System health monitoring
    - Execution quality monitoring
    
    Note: Does NOT handle:
    - Cross-model validation (Tactician already handles Analyst input)
    - Signal quality validation before execution (removed)
    - Correlation limits (removed)
    - Single-asset position sizing (handled elsewhere)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Trading Supervisor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('TradingSupervisor')

        # Extract trading config
        self.trading_config = TradingConfig(**config.get('trading_config', {}))
        
        # Supervisor configuration
        supervisor_config = config.get('supervisor', {})
        
        # Risk oversight settings
        self.max_portfolio_risk = supervisor_config.get('max_portfolio_risk', 0.02)
        self.max_drawdown = supervisor_config.get('max_drawdown', 0.15)
        self.max_total_exposure = supervisor_config.get('max_total_exposure', 1.0)  # 100% of portfolio
        
        # Cross-asset limits (to avoid over-correlation)
        self.max_cross_asset_exposure = supervisor_config.get('max_cross_asset_exposure', 0.5)  # Max 50% in correlated assets
        self.cross_asset_correlation_threshold = supervisor_config.get('cross_asset_correlation_threshold', 0.7)
        
        # Circuit breaker settings
        circuit_breaker_config = supervisor_config.get('circuit_breakers', {})
        self.circuit_breakers_enabled = circuit_breaker_config.get('enabled', True)
        self.max_loss_per_hour = circuit_breaker_config.get('max_loss_per_hour', 0.05)  # 5% max loss per hour
        self.max_loss_per_day = circuit_breaker_config.get('max_loss_per_day', 0.10)  # 10% max loss per day
        self.max_rejections_per_minute = circuit_breaker_config.get('max_rejections_per_minute', 5)
        self.max_slippage_per_trade = circuit_breaker_config.get('max_slippage_per_trade', 0.005)  # 0.5%
        self.circuit_breaker_cooldown = circuit_breaker_config.get('cooldown_period_seconds', 300)  # 5 minutes
        
        # Execution quality settings
        execution_config = supervisor_config.get('execution_quality', {})
        self.min_fill_rate = execution_config.get('min_fill_rate', 0.95)  # 95% orders must fill
        self.max_avg_slippage = execution_config.get('max_avg_slippage', 0.002)  # 0.2% max average slippage
        self.track_execution_metrics = execution_config.get('track_execution_metrics', True)
        
        # System health settings
        self.monitor_data_quality = supervisor_config.get('monitor_data_quality', True)
        self.monitor_exchange_health = supervisor_config.get('monitor_exchange_health', True)
        
        # State management
        self.status = SupervisorStatus.INITIALIZING
        self.is_initialized = False
        
        # Circuit breaker state
        self.circuit_breaker = CircuitBreakerState()
        
        # Portfolio tracking
        self.all_active_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position dict
        self.total_portfolio_exposure: float = 0.0
        self.total_portfolio_risk: float = 0.0
        
        # Performance tracking
        self.hourly_losses: List[Tuple[datetime, float]] = []  # (timestamp, loss)
        self.daily_losses: List[Tuple[datetime, float]] = []  # (timestamp, loss)
        self.recent_rejections: List[datetime] = []  # List of rejection timestamps
        
        # Execution quality tracking
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'total_slippage': 0.0,
            'total_commissions': 0.0,
            'recent_executions': []  # Last 100 executions
        }
        
        # Cross-asset correlation tracking
        self.cross_asset_exposure: Dict[str, float] = {}  # Asset group -> exposure
        self.correlated_asset_groups = supervisor_config.get('correlated_asset_groups', {
            'crypto_majors': ['BTCUSDT', 'ETHUSDT'],
            'crypto_altcoins': ['SOLUSDT', 'ADAUSDT', 'DOTUSDT'],
            # Add more groups as needed
        })
        
        # Risk calculator for portfolio-level calculations
        self.risk_calculator = RiskCalculator(self.trading_config)
        
        # Reference to orchestrator (set during integration)
        self.orchestrator_reference: Optional[Any] = None
        
        tprint_info("🚀 Initializing Trading Supervisor...")

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.CRITICAL,
        raise_on_error=True
    )
    async def initialize(self) -> bool:
        """
        Initialize the Trading Supervisor.

        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize risk calculator
            await self.risk_calculator.initialize()
            
            # Initialize circuit breaker state
            self.circuit_breaker = CircuitBreakerState()
            
            # Clear tracking structures
            self.all_active_positions.clear()
            self.hourly_losses.clear()
            self.daily_losses.clear()
            self.recent_rejections.clear()
            self.execution_stats.clear()
            self.cross_asset_exposure.clear()
            
            self.status = SupervisorStatus.ACTIVE
            self.is_initialized = True
            
            tprint_success("✅ Trading Supervisor initialized successfully")
            self.logger.info("Trading Supervisor initialized with configuration:")
            self.logger.info(f"  - Max portfolio risk: {self.max_portfolio_risk}")
            self.logger.info(f"  - Max drawdown: {self.max_drawdown}")
            self.logger.info(f"  - Max total exposure: {self.max_total_exposure}")
            self.logger.info(f"  - Max cross-asset exposure: {self.max_cross_asset_exposure}")
            self.logger.info(f"  - Circuit breakers enabled: {self.circuit_breakers_enabled}")
            
            return True

        except Exception as e:
            self.status = SupervisorStatus.ERROR
            self.logger.error(f"❌ Failed to initialize Trading Supervisor: {e}")
            tprint_error(f"❌ Failed to initialize Trading Supervisor: {e}")
            raise

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def pre_decision_validation(
        self,
        symbol: str,
        current_positions: Dict[str, Dict[str, Any]],
        market_snapshot: Optional[Dict[str, Any]] = None,
        account_balance: Optional[float] = None
    ) -> ValidationResult:
        """
        Pre-decision validation - checks before signal generation.
        
        Validates:
        - Circuit breaker status
        - Portfolio-level risk limits
        - System health
        
        Args:
            symbol: Trading symbol
            current_positions: Current positions for this symbol
            market_snapshot: Optional market data snapshot
            account_balance: Optional account balance
            
        Returns:
            ValidationResult: Validation result
        """
        if not self.is_initialized:
            return ValidationResult(
                is_valid=False,
                reasons=["Supervisor not initialized"],
                risk_score=1.0
            )
        
        reasons = []
        warnings = []
        risk_score = 0.0
        
        # Check circuit breaker
        if self.circuit_breaker.triggered:
            if self.circuit_breaker.cooldown_until and datetime.now() < self.circuit_breaker.cooldown_until:
                return ValidationResult(
                    is_valid=False,
                    reasons=[f"Circuit breaker active: {self.circuit_breaker.trigger_reason}"],
                    risk_score=1.0
                )
            else:
                # Circuit breaker cooldown expired, reset
                self.circuit_breaker.triggered = False
                self.circuit_breaker.trigger_time = None
                self.circuit_breaker.cooldown_until = None
                tprint_info("✅ Circuit breaker cooldown expired, resetting")
        
        # Check portfolio-level risk limits
        portfolio_risk_valid, risk_reasons = await self._check_portfolio_risk_limits(
            symbol, current_positions, account_balance
        )
        if not portfolio_risk_valid:
            reasons.extend(risk_reasons)
            risk_score += 0.5
        
        # Check total exposure limits
        exposure_valid, exposure_reasons = await self._check_total_exposure_limits(
            current_positions, account_balance
        )
        if not exposure_valid:
            reasons.extend(exposure_reasons)
            risk_score += 0.3
        
        # System health checks
        if market_snapshot:
            health_valid, health_reasons = await self._check_system_health(market_snapshot)
            if not health_valid:
                warnings.extend(health_reasons)
        
        is_valid = len(reasons) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            reasons=reasons,
            warnings=warnings,
            risk_score=min(risk_score, 1.0),
            metadata={
                'circuit_breaker_status': self.circuit_breaker.triggered,
                'portfolio_risk': self.total_portfolio_risk,
                'total_exposure': self.total_portfolio_exposure
            }
        )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def validate_decision(
        self,
        decision: Any,  # TradingDecision
        analyst_signal: Optional[Any] = None,
        tactician_signal: Optional[Any] = None,
        combined_signal: Optional[Dict[str, Any]] = None,
        current_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        account_balance: Optional[float] = None
    ) -> DecisionApproval:
        """
        Validate trading decision after signal generation.
        
        Note: Does NOT perform cross-model validation (Tactician already handles Analyst input).
        Focuses on portfolio-level and cross-asset risk checks.
        
        Args:
            decision: TradingDecision object
            analyst_signal: Optional Analyst signal (for reference, not validation)
            tactician_signal: Optional Tactician signal (for reference, not validation)
            combined_signal: Optional combined signal dict
            current_positions: Current active positions
            account_balance: Account balance
            
        Returns:
            DecisionApproval: Approval result
        """
        if not self.is_initialized:
            return DecisionApproval(
                approved=False,
                reason="Supervisor not initialized",
                confidence_modifier=0.0
            )
        
        # Check cross-asset exposure limits
        cross_asset_valid, cross_asset_reason = await self._check_cross_asset_exposure(
            decision, current_positions, account_balance
        )
        
        if not cross_asset_valid:
            return DecisionApproval(
                approved=False,
                reason=cross_asset_reason,
                confidence_modifier=0.0,
                metadata={'validation_type': 'cross_asset_exposure'}
            )
        
        # Check portfolio-level risk with new decision
        portfolio_valid, portfolio_reason = await self._check_portfolio_risk_with_decision(
            decision, current_positions, account_balance
        )
        
        if not portfolio_valid:
            return DecisionApproval(
                approved=False,
                reason=portfolio_reason,
                confidence_modifier=0.0,
                metadata={'validation_type': 'portfolio_risk'}
            )
        
        # All checks passed
        return DecisionApproval(
            approved=True,
            reason="All supervisor validations passed",
            confidence_modifier=1.0,
            metadata={
                'validated_at': datetime.now().isoformat(),
                'portfolio_risk': self.total_portfolio_risk,
                'cross_asset_exposure': self._calculate_cross_asset_exposure()
            }
        )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.CRITICAL,
        raise_on_error=False
    )
    async def pre_execution_check(
        self,
        decision: Any,  # TradingDecision
        current_exposure: float,
        risk_metrics: Dict[str, float],
        account_balance: Optional[float] = None
    ) -> ExecutionCheck:
        """
        Final check before order execution.
        
        This is the last safety check before actual order placement.
        Focuses on execution-level risks and circuit breakers.
        
        Args:
            decision: TradingDecision object
            current_exposure: Current total portfolio exposure
            risk_metrics: Risk metrics for this trade
            account_balance: Account balance
            
        Returns:
            ExecutionCheck: Check result
        """
        if not self.is_initialized:
            return ExecutionCheck(
                can_proceed=False,
                reason="Supervisor not initialized"
            )
        
        reasons = []
        suggested_adjustments = {}
        
        # Check circuit breaker (critical - must pass)
        if self.circuit_breaker.triggered:
            if self.circuit_breaker.cooldown_until and datetime.now() < self.circuit_breaker.cooldown_until:
                return ExecutionCheck(
                    can_proceed=False,
                    reason=f"Circuit breaker active: {self.circuit_breaker.trigger_reason}",
                    metadata={'circuit_breaker_triggered': True}
                )
        
        # Check total exposure limit (portfolio-level)
        if current_exposure > self.max_total_exposure:
            reasons.append(f"Total exposure {current_exposure:.2%} exceeds limit {self.max_total_exposure:.2%}")
            suggested_adjustments['reduce_position_size'] = True
            suggested_adjustments['target_exposure'] = self.max_total_exposure * 0.9
        
        # Check execution quality (if we have recent poor execution stats)
        if self.track_execution_metrics:
            quality_check = self._check_execution_quality_trends()
            if not quality_check['acceptable']:
                reasons.append(f"Execution quality below threshold: {quality_check['reason']}")
                if quality_check.get('suggest_reduce_size'):
                    suggested_adjustments['reduce_position_size'] = True
        
        # Check recent rejection rate
        if len(self.recent_rejections) >= self.max_rejections_per_minute:
            return ExecutionCheck(
                can_proceed=False,
                reason=f"Too many rejections in last minute: {len(self.recent_rejections)}",
                metadata={'rejection_count': len(self.recent_rejections)}
            )
        
        can_proceed = len(reasons) == 0
        
        return ExecutionCheck(
            can_proceed=can_proceed,
            reason="; ".join(reasons) if reasons else "All pre-execution checks passed",
            suggested_adjustments=suggested_adjustments,
            metadata={
                'current_exposure': current_exposure,
                'max_exposure': self.max_total_exposure,
                'circuit_breaker_status': self.circuit_breaker.triggered
            }
        )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def monitor_execution(
        self,
        order_id: str,
        execution_result: Dict[str, Any]
    ) -> None:
        """
        Monitor order execution quality.
        
        Args:
            order_id: Order ID
            execution_result: Execution result dictionary
        """
        try:
            self.execution_stats['total_orders'] += 1
            
            status = execution_result.get('status', '').upper()
            if status in ['FILLED', 'PARTIALLY_FILLED']:
                self.execution_stats['filled_orders'] += 1
                
                # Track slippage if available
                if 'slippage' in execution_result:
                    slippage = abs(execution_result['slippage'])
                    self.execution_stats['total_slippage'] += slippage
                    
                    # Check for excessive slippage
                    if slippage > self.max_slippage_per_trade:
                        tprint_warning(
                            f"⚠️ High slippage detected: {slippage:.4%} for order {order_id}"
                        )
                    
                    # Update recent executions
                    if len(self.execution_stats['recent_executions']) >= 100:
                        self.execution_stats['recent_executions'].pop(0)
                    self.execution_stats['recent_executions'].append({
                        'order_id': order_id,
                        'slippage': slippage,
                        'timestamp': datetime.now()
                    })
                    
            elif status in ['REJECTED', 'CANCELLED']:
                self.execution_stats['rejected_orders'] += 1
                self.recent_rejections.append(datetime.now())
                
                # Clean old rejections (older than 1 minute)
                cutoff = datetime.now() - timedelta(minutes=1)
                self.recent_rejections = [r for r in self.recent_rejections if r > cutoff]
                
                # Check circuit breaker trigger
                if len(self.recent_rejections) >= self.max_rejections_per_minute:
                    await self._trigger_circuit_breaker(
                        f"Too many rejections: {len(self.recent_rejections)} in last minute"
                    )
            
            # Track commission if available
            if 'commission' in execution_result:
                self.execution_stats['total_commissions'] += execution_result['commission']
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error monitoring execution: {e}")

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def post_trade_analysis(
        self,
        trade_id: str,
        trade_outcome: Dict[str, Any]
    ) -> None:
        """
        Analyze completed trade for quality and circuit breaker triggers.
        
        Args:
            trade_id: Trade ID
            trade_outcome: Trade outcome dictionary with PnL, duration, etc.
        """
        try:
            # Extract PnL if available
            pnl = trade_outcome.get('pnl_absolute', 0.0)
            if pnl < 0:  # Loss
                loss_amount = abs(pnl)
                now = datetime.now()
                
                # Track hourly losses
                self.hourly_losses.append((now, loss_amount))
                cutoff = now - timedelta(hours=1)
                self.hourly_losses = [(t, l) for t, l in self.hourly_losses if t > cutoff]
                
                # Track daily losses
                self.daily_losses.append((now, loss_amount))
                daily_cutoff = now - timedelta(days=1)
                self.daily_losses = [(t, l) for t, l in self.daily_losses if t > daily_cutoff]
                
                # Check circuit breakers
                if self.circuit_breakers_enabled:
                    await self._check_loss_based_circuit_breakers()
            
            # Update execution quality metrics
            if 'slippage' in trade_outcome:
                await self.monitor_execution(
                    trade_id,
                    {'status': 'FILLED', 'slippage': trade_outcome['slippage']}
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error in post-trade analysis: {e}")

    @handles_errors
    async def _check_portfolio_risk_limits(
        self,
        symbol: str,
        current_positions: Dict[str, Dict[str, Any]],
        account_balance: Optional[float]
    ) -> Tuple[bool, List[str]]:
        """Check portfolio-level risk limits."""
        if not account_balance or account_balance <= 0:
            return True, []  # Skip check if balance unavailable
        
        reasons = []
        
        # Calculate total portfolio risk
        total_risk = 0.0
        for pos_id, position in current_positions.items():
            pos_value = position.get('quantity', 0) * position.get('entry_price', 0)
            pos_risk = pos_value / account_balance
            total_risk += pos_risk
        
        self.total_portfolio_risk = total_risk
        
        if total_risk > self.max_portfolio_risk:
            reasons.append(
                f"Portfolio risk {total_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
            )
        
        return len(reasons) == 0, reasons

    @handles_errors
    async def _check_total_exposure_limits(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        account_balance: Optional[float]
    ) -> Tuple[bool, List[str]]:
        """Check total portfolio exposure limits."""
        if not account_balance or account_balance <= 0:
            return True, []  # Skip check if balance unavailable
        
        reasons = []
        
        # Calculate total exposure
        total_exposure = 0.0
        for pos_id, position in current_positions.items():
            pos_value = position.get('quantity', 0) * position.get('entry_price', 0)
            leverage = position.get('leverage', 1.0)
            exposure = (pos_value * leverage) / account_balance
            total_exposure += exposure
        
        self.total_portfolio_exposure = total_exposure
        
        if total_exposure > self.max_total_exposure:
            reasons.append(
                f"Total exposure {total_exposure:.2%} exceeds limit {self.max_total_exposure:.2%}"
            )
        
        return len(reasons) == 0, reasons

    @handles_errors
    async def _check_cross_asset_exposure(
        self,
        decision: Any,
        current_positions: Optional[Dict[str, Dict[str, Any]]],
        account_balance: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Check cross-asset exposure limits to avoid over-correlation.
        
        This checks that we don't have too much exposure in correlated asset groups.
        Single-asset limits are handled elsewhere.
        """
        if not account_balance or account_balance <= 0:
            return True, ""  # Skip check if balance unavailable
        
        if not current_positions:
            current_positions = {}
        
        # Calculate exposure per asset group
        symbol = getattr(decision, 'symbol', '')
        decision_quantity = getattr(decision, 'quantity', 0)
        decision_price = getattr(decision, 'price', 0)
        decision_value = decision_quantity * decision_price
        
        # Find which asset group this symbol belongs to
        symbol_group = None
        for group_name, symbols in self.correlated_asset_groups.items():
            if symbol in symbols:
                symbol_group = group_name
                break
        
        if not symbol_group:
            # Symbol not in any correlated group - allow
            return True, ""
        
        # Calculate current exposure for this asset group
        current_group_exposure = 0.0
        
        # Add existing positions in this group
        for pos_id, position in current_positions.items():
            pos_symbol = position.get('symbol', '')
            if pos_symbol in self.correlated_asset_groups.get(symbol_group, []):
                pos_value = position.get('quantity', 0) * position.get('entry_price', 0)
                leverage = position.get('leverage', 1.0)
                exposure = (pos_value * leverage) / account_balance
                current_group_exposure += exposure
        
        # Add new decision exposure
        decision_exposure = decision_value / account_balance
        new_group_exposure = current_group_exposure + decision_exposure
        
        # Check against limit
        if new_group_exposure > self.max_cross_asset_exposure:
            return False, (
                f"Cross-asset exposure for group '{symbol_group}' would be {new_group_exposure:.2%}, "
                f"exceeding limit {self.max_cross_asset_exposure:.2%}. "
                f"Current exposure: {current_group_exposure:.2%}"
            )
        
        # Update tracking
        self.cross_asset_exposure[symbol_group] = new_group_exposure
        
        return True, ""

    @handles_errors
    async def _check_portfolio_risk_with_decision(
        self,
        decision: Any,
        current_positions: Optional[Dict[str, Dict[str, Any]]],
        account_balance: Optional[float]
    ) -> Tuple[bool, str]:
        """Check portfolio risk including the new decision."""
        if not account_balance or account_balance <= 0:
            return True, ""  # Skip check if balance unavailable
        
        if not current_positions:
            current_positions = {}
        
        # Calculate total risk including new decision
        total_risk = self.total_portfolio_risk
        
        # Add risk from new decision
        decision_quantity = getattr(decision, 'quantity', 0)
        decision_price = getattr(decision, 'price', 0)
        decision_value = decision_quantity * decision_price
        decision_risk = decision_value / account_balance
        
        new_total_risk = total_risk + decision_risk
        
        if new_total_risk > self.max_portfolio_risk:
            return False, (
                f"Portfolio risk with new decision would be {new_total_risk:.2%}, "
                f"exceeding limit {self.max_portfolio_risk:.2%}"
            )
        
        return True, ""

    @handles_errors
    async def _check_system_health(
        self,
        market_snapshot: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Check system health (data quality, exchange connectivity)."""
        warnings = []
        
        if self.monitor_data_quality:
            # Check data freshness if timestamp available
            if 'timestamp' in market_snapshot:
                data_age = (datetime.now() - market_snapshot['timestamp']).total_seconds()
                if data_age > 300:  # 5 minutes
                    warnings.append(f"Market data is {data_age:.0f}s old")
        
        # Add more health checks as needed
        
        return len(warnings) == 0, warnings

    def _check_execution_quality_trends(self) -> Dict[str, Any]:
        """Check execution quality trends."""
        stats = self.execution_stats
        
        if stats['total_orders'] == 0:
            return {'acceptable': True, 'reason': 'No execution history'}
        
        # Calculate fill rate
        fill_rate = stats['filled_orders'] / stats['total_orders']
        
        if fill_rate < self.min_fill_rate:
            return {
                'acceptable': False,
                'reason': f"Fill rate {fill_rate:.2%} below threshold {self.min_fill_rate:.2%}",
                'suggest_reduce_size': True
            }
        
        # Calculate average slippage
        if len(stats['recent_executions']) > 0:
            avg_slippage = stats['total_slippage'] / len(stats['recent_executions'])
            if avg_slippage > self.max_avg_slippage:
                return {
                    'acceptable': False,
                    'reason': f"Average slippage {avg_slippage:.4%} exceeds threshold {self.max_avg_slippage:.4%}",
                    'suggest_reduce_size': True
                }
        
        return {'acceptable': True, 'reason': 'Execution quality acceptable'}

    async def _check_loss_based_circuit_breakers(self) -> None:
        """Check if loss-based circuit breakers should trigger."""
        if not self.circuit_breakers_enabled:
            return
        
        # Calculate hourly loss
        if self.hourly_losses:
            hourly_loss_total = sum(loss for _, loss in self.hourly_losses)
            # Assuming we have account balance somewhere
            # For now, check absolute loss thresholds
            if hourly_loss_total > 0:  # We'd need account_balance to calculate percentage
                # This would need account_balance to properly calculate
                pass
        
        # Calculate daily loss
        if self.daily_losses:
            daily_loss_total = sum(loss for _, loss in self.daily_losses)
            if daily_loss_total > 0:
                # This would need account_balance to properly calculate percentage
                pass

    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker."""
        if not self.circuit_breakers_enabled:
            return
        
        self.circuit_breaker.triggered = True
        self.circuit_breaker.trigger_time = datetime.now()
        self.circuit_breaker.trigger_reason = reason
        self.circuit_breaker.cooldown_until = datetime.now() + timedelta(seconds=self.circuit_breaker_cooldown)
        self.circuit_breaker.trigger_count += 1
        
        self.status = SupervisorStatus.CIRCUIT_BREAKER_TRIGGERED
        
        tprint_error(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        tprint_error(f"⏸️ Trading paused for {self.circuit_breaker_cooldown}s")
        self.logger.critical(f"Circuit breaker triggered: {reason}")
        self.logger.critical(f"Cooldown until: {self.circuit_breaker.cooldown_until}")

    def _calculate_cross_asset_exposure(self) -> Dict[str, float]:
        """Calculate current cross-asset exposure per group."""
        return self.cross_asset_exposure.copy()

    async def update_positions(
        self,
        positions_by_symbol: Dict[str, Dict[str, Any]],
        account_balance: float
    ) -> None:
        """
        Update supervisor's view of all active positions across all symbols.
        
        This allows the Supervisor to track portfolio-level exposure.
        
        Args:
            positions_by_symbol: Dict of symbol -> positions dict
            account_balance: Current account balance
        """
        try:
            self.all_active_positions = positions_by_symbol.copy()
            
            # Recalculate portfolio metrics
            total_exposure = 0.0
            total_risk = 0.0
            
            for symbol, positions in positions_by_symbol.items():
                # Aggregate positions for symbol (if multiple)
                symbol_value = 0.0
                symbol_risk = 0.0
                
                # Positions dict may contain multiple position entries
                if isinstance(positions, dict):
                    # Check if it's a single position dict or multiple
                    if 'quantity' in positions:
                        # Single position
                        symbol_value = positions.get('quantity', 0) * positions.get('entry_price', 0)
                        leverage = positions.get('leverage', 1.0)
                        symbol_risk = symbol_value / account_balance
                        total_exposure += (symbol_value * leverage) / account_balance
                        total_risk += symbol_risk
                    else:
                        # Multiple positions keyed by position_id
                        for pos_id, position in positions.items():
                            if isinstance(position, dict) and 'quantity' in position:
                                pos_value = position.get('quantity', 0) * position.get('entry_price', 0)
                                leverage = position.get('leverage', 1.0)
                                pos_risk = pos_value / account_balance
                                total_exposure += (pos_value * leverage) / account_balance
                                total_risk += pos_risk
            
            self.total_portfolio_exposure = total_exposure
            self.total_portfolio_risk = total_risk
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating positions: {e}")

    def get_supervisor_status(self) -> Dict[str, Any]:
        """Get current supervisor status and metrics."""
        return {
            'status': self.status.value,
            'is_initialized': self.is_initialized,
            'circuit_breaker': {
                'triggered': self.circuit_breaker.triggered,
                'trigger_reason': self.circuit_breaker.trigger_reason,
                'trigger_count': self.circuit_breaker.trigger_count,
                'cooldown_until': self.circuit_breaker.cooldown_until.isoformat() if self.circuit_breaker.cooldown_until else None
            },
            'portfolio_metrics': {
                'total_portfolio_risk': self.total_portfolio_risk,
                'total_portfolio_exposure': self.total_portfolio_exposure,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_total_exposure': self.max_total_exposure
            },
            'cross_asset_exposure': self.cross_asset_exposure.copy(),
            'execution_stats': {
                'total_orders': self.execution_stats['total_orders'],
                'filled_orders': self.execution_stats['filled_orders'],
                'rejected_orders': self.execution_stats['rejected_orders'],
                'fill_rate': (
                    self.execution_stats['filled_orders'] / self.execution_stats['total_orders']
                    if self.execution_stats['total_orders'] > 0 else 0.0
                ),
                'avg_slippage': (
                    self.execution_stats['total_slippage'] / len(self.execution_stats['recent_executions'])
                    if len(self.execution_stats['recent_executions']) > 0 else 0.0
                )
            },
            'recent_rejections_count': len(self.recent_rejections)
        }

    async def stop(self) -> None:
        """Stop the supervisor."""
        try:
            self.status = SupervisorStatus.DISABLED
            self.is_initialized = False
            tprint_info("🛑 Trading Supervisor stopped")
            self.logger.info("Trading Supervisor stopped")
        except Exception as e:
            self.logger.error(f"❌ Error stopping supervisor: {e}")


# Factory function
def create_trading_supervisor(config: Dict[str, Any]) -> TradingSupervisor:
    """Create a Trading Supervisor instance."""
    return TradingSupervisor(config)
