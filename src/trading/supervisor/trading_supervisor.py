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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
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

if TYPE_CHECKING:
    from ..execution.trading_orchestrator import TradingDecision

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

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Trading Supervisor.

        Args:
            config: Configuration dictionary
        """
        tprint_info("🚀 Initializing Trading Supervisor...")
        tprint_structured(
            "Supervisor Configuration",
            {
                "phase": "initialization",
                "config_keys": list(config.keys())
            },
            level=LogLevel.INFO
        )
        
        self.config: Dict[str, Any] = config
        self.logger = logger.getChild('TradingSupervisor')

        # Extract trading config
        self.trading_config = TradingConfig(**config.get('trading_config', {}))
        
        # Supervisor configuration
        supervisor_config: Dict[str, Any] = config.get('supervisor', {})
        
        # Risk oversight settings
        self.max_portfolio_risk: float = supervisor_config.get('max_portfolio_risk', 0.02)
        self.max_drawdown: float = supervisor_config.get('max_drawdown', 0.15)
        self.max_total_exposure: float = supervisor_config.get('max_total_exposure', 1.0)  # 100% of portfolio
        
        # Cross-asset limits (to avoid over-correlation)
        self.max_cross_asset_exposure: float = supervisor_config.get('max_cross_asset_exposure', 0.5)  # Max 50% in correlated assets
        self.cross_asset_correlation_threshold: float = supervisor_config.get('cross_asset_correlation_threshold', 0.7)
        
        # Circuit breaker settings
        circuit_breaker_config: Dict[str, Any] = supervisor_config.get('circuit_breakers', {})
        self.circuit_breakers_enabled: bool = circuit_breaker_config.get('enabled', True)
        self.max_loss_per_hour: float = circuit_breaker_config.get('max_loss_per_hour', 0.05)  # 5% max loss per hour
        self.max_loss_per_day: float = circuit_breaker_config.get('max_loss_per_day', 0.10)  # 10% max loss per day
        self.max_rejections_per_minute: int = circuit_breaker_config.get('max_rejections_per_minute', 5)
        self.max_slippage_per_trade: float = circuit_breaker_config.get('max_slippage_per_trade', 0.005)  # 0.5%
        self.circuit_breaker_cooldown: int = circuit_breaker_config.get('cooldown_period_seconds', 300)  # 5 minutes
        
        # Execution quality settings
        execution_config: Dict[str, Any] = supervisor_config.get('execution_quality', {})
        self.min_fill_rate: float = execution_config.get('min_fill_rate', 0.95)  # 95% orders must fill
        self.max_avg_slippage: float = execution_config.get('max_avg_slippage', 0.002)  # 0.2% max average slippage
        self.track_execution_metrics: bool = execution_config.get('track_execution_metrics', True)
        
        # System health settings
        self.monitor_data_quality: bool = supervisor_config.get('monitor_data_quality', True)
        self.monitor_exchange_health: bool = supervisor_config.get('monitor_exchange_health', True)
        
        # State management
        self.status: SupervisorStatus = SupervisorStatus.INITIALIZING
        self.is_initialized: bool = False
        
        # Thread safety locks
        self._circuit_breaker_lock: asyncio.Lock = asyncio.Lock()
        self._positions_lock: asyncio.Lock = asyncio.Lock()
        self._execution_stats_lock: asyncio.Lock = asyncio.Lock()
        
        # Circuit breaker state
        self.circuit_breaker: CircuitBreakerState = CircuitBreakerState()
        
        # Account balance tracking
        self.account_balance: Optional[float] = None
        
        # Portfolio tracking
        self.all_active_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position dict
        self.total_portfolio_exposure: float = 0.0
        self.total_portfolio_risk: float = 0.0
        
        # Performance tracking
        self.hourly_losses: List[Tuple[datetime, float]] = []  # (timestamp, loss)
        self.daily_losses: List[Tuple[datetime, float]] = []  # (timestamp, loss)
        self.recent_rejections: List[datetime] = []  # List of rejection timestamps
        
        # Execution quality tracking
        self.execution_stats: Dict[str, Any] = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'total_slippage': 0.0,
            'total_commissions': 0.0,
            'recent_executions': []  # Last 100 executions
        }
        
        # Cross-asset correlation tracking
        self.cross_asset_exposure: Dict[str, float] = {}  # Asset group -> exposure
        self.correlated_asset_groups: Dict[str, List[str]] = supervisor_config.get('correlated_asset_groups', {
            'crypto_majors': ['BTCUSDT', 'ETHUSDT'],
            'crypto_altcoins': ['SOLUSDT', 'ADAUSDT', 'DOTUSDT'],
            # Add more groups as needed
        })
        
        # Risk calculator for portfolio-level calculations
        self.risk_calculator: RiskCalculator = RiskCalculator(self.trading_config)
        
        # Reference to orchestrator (set during integration)
        self.orchestrator_reference: Optional[Any] = None
        
        tprint_info(f"✓ Supervisor initialized with risk limits: portfolio={self.max_portfolio_risk:.2%}, exposure={self.max_total_exposure:.2%}")

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
        tprint_info("🔄 Starting Supervisor initialization...")
        try:
            # Initialize risk calculator
            tprint_info("📊 Initializing risk calculator...")
            await self.risk_calculator.initialize()
            tprint_success("✓ Risk calculator initialized")
            
            # Initialize circuit breaker state
            self.circuit_breaker = CircuitBreakerState()
            tprint_info(f"🛡️ Circuit breakers: {'enabled' if self.circuit_breakers_enabled else 'disabled'}")
            
            # Clear tracking structures
            self.all_active_positions.clear()
            self.hourly_losses.clear()
            self.daily_losses.clear()
            self.recent_rejections.clear()
            self.execution_stats.clear()
            self.cross_asset_exposure.clear()
            tprint_info("✓ Tracking structures cleared")
            
            # Initialize account balance from config if available
            self.account_balance = self.config.get('account_balance')
            if self.account_balance:
                tprint_info(f"💰 Initial account balance: {self.account_balance:.2f}")
            
            self.status = SupervisorStatus.ACTIVE
            self.is_initialized = True
            
            tprint_success("✅ Trading Supervisor initialized successfully")
            tprint_structured(
                "Supervisor Configuration",
                {
                    "max_portfolio_risk": f"{self.max_portfolio_risk:.2%}",
                    "max_drawdown": f"{self.max_drawdown:.2%}",
                    "max_total_exposure": f"{self.max_total_exposure:.2%}",
                    "max_cross_asset_exposure": f"{self.max_cross_asset_exposure:.2%}",
                    "circuit_breakers_enabled": self.circuit_breakers_enabled
                },
                level=LogLevel.INFO
            )
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
        tprint_info(f"🔍 Pre-decision validation for {symbol}...")
        
        if not self.is_initialized:
            tprint_warning(f"⚠️ Supervisor not initialized, validation failed for {symbol}")
            return ValidationResult(
                is_valid=False,
                reasons=["Supervisor not initialized"],
                risk_score=1.0
            )
        
        reasons: List[str] = []
        warnings: List[str] = []
        risk_score: float = 0.0
        
        # Check circuit breaker (with thread safety)
        async with self._circuit_breaker_lock:
            if self.circuit_breaker.triggered:
                if self.circuit_breaker.cooldown_until and datetime.now() < self.circuit_breaker.cooldown_until:
                    tprint_warning(f"🚨 Circuit breaker active for {symbol}: {self.circuit_breaker.trigger_reason}")
                    return ValidationResult(
                        is_valid=False,
                        reasons=[f"Circuit breaker active: {self.circuit_breaker.trigger_reason}"],
                        risk_score=1.0
                    )
                else:
                    # Circuit breaker cooldown expired, reset atomically
                    self.circuit_breaker.triggered = False
                    self.circuit_breaker.trigger_time = None
                    self.circuit_breaker.cooldown_until = None
                    self.status = SupervisorStatus.ACTIVE
                    tprint_info("✅ Circuit breaker cooldown expired, resetting")
        
        # Check portfolio-level risk limits
        tprint_info(f"📊 Checking portfolio risk limits for {symbol}...")
        portfolio_risk_valid, risk_reasons = await self._check_portfolio_risk_limits(
            symbol, current_positions, account_balance
        )
        if not portfolio_risk_valid:
            reasons.extend(risk_reasons)
            risk_score += 0.5
            tprint_warning(f"⚠️ Portfolio risk check failed for {symbol}: {'; '.join(risk_reasons)}")
        else:
            tprint_info(f"✓ Portfolio risk check passed for {symbol}")
        
        # Check total exposure limits
        tprint_info(f"📈 Checking total exposure limits for {symbol}...")
        exposure_valid, exposure_reasons = await self._check_total_exposure_limits(
            current_positions, account_balance
        )
        if not exposure_valid:
            reasons.extend(exposure_reasons)
            risk_score += 0.3
            tprint_warning(f"⚠️ Exposure limit check failed for {symbol}: {'; '.join(exposure_reasons)}")
        else:
            tprint_info(f"✓ Exposure limit check passed for {symbol}")
        
        # System health checks
        if market_snapshot:
            tprint_info(f"🏥 Checking system health for {symbol}...")
            health_valid, health_reasons = await self._check_system_health(market_snapshot)
            if not health_valid:
                warnings.extend(health_reasons)
                tprint_warning(f"⚠️ System health warnings for {symbol}: {'; '.join(health_reasons)}")
            else:
                tprint_info(f"✓ System health check passed for {symbol}")
        
        is_valid: bool = len(reasons) == 0
        
        if is_valid:
            tprint_success(f"✅ Pre-decision validation passed for {symbol}")
        else:
            tprint_error(f"❌ Pre-decision validation failed for {symbol}: {'; '.join(reasons)}")
        
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
        decision: "TradingDecision",
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
        symbol: str = getattr(decision, 'symbol', 'UNKNOWN')
        action: str = getattr(decision, 'action', 'UNKNOWN')
        tprint_info(f"🔍 Validating decision for {symbol} ({action})...")
        
        if not self.is_initialized:
            tprint_warning(f"⚠️ Supervisor not initialized, decision validation failed for {symbol}")
            return DecisionApproval(
                approved=False,
                reason="Supervisor not initialized",
                confidence_modifier=0.0
            )
        
        # Check cross-asset exposure limits
        tprint_info(f"📊 Checking cross-asset exposure for {symbol}...")
        cross_asset_valid, cross_asset_reason = await self._check_cross_asset_exposure(
            decision, current_positions, account_balance
        )
        
        if not cross_asset_valid:
            tprint_warning(f"⚠️ Cross-asset exposure check failed for {symbol}: {cross_asset_reason}")
            return DecisionApproval(
                approved=False,
                reason=cross_asset_reason,
                confidence_modifier=0.0,
                metadata={'validation_type': 'cross_asset_exposure'}
            )
        else:
            tprint_info(f"✓ Cross-asset exposure check passed for {symbol}")
        
        # Check portfolio-level risk with new decision
        tprint_info(f"📈 Checking portfolio risk with decision for {symbol}...")
        portfolio_valid, portfolio_reason = await self._check_portfolio_risk_with_decision(
            decision, current_positions, account_balance
        )
        
        if not portfolio_valid:
            tprint_warning(f"⚠️ Portfolio risk check failed for {symbol}: {portfolio_reason}")
            return DecisionApproval(
                approved=False,
                reason=portfolio_reason,
                confidence_modifier=0.0,
                metadata={'validation_type': 'portfolio_risk'}
            )
        else:
            tprint_info(f"✓ Portfolio risk check passed for {symbol}")
        
        # All checks passed
        tprint_success(f"✅ Decision validation passed for {symbol} ({action})")
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
        decision: "TradingDecision",
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
        symbol: str = getattr(decision, 'symbol', 'UNKNOWN')
        action: str = getattr(decision, 'action', 'UNKNOWN')
        tprint_info(f"🔍 Pre-execution check for {symbol} ({action})...")
        
        if not self.is_initialized:
            tprint_warning(f"⚠️ Supervisor not initialized, pre-execution check failed for {symbol}")
            return ExecutionCheck(
                can_proceed=False,
                reason="Supervisor not initialized"
            )
        
        reasons: List[str] = []
        suggested_adjustments: Dict[str, Any] = {}
        
        # Check circuit breaker (critical - must pass, with thread safety)
        async with self._circuit_breaker_lock:
            if self.circuit_breaker.triggered:
                if self.circuit_breaker.cooldown_until and datetime.now() < self.circuit_breaker.cooldown_until:
                    tprint_error(f"🚨 Circuit breaker active, blocking execution for {symbol}: {self.circuit_breaker.trigger_reason}")
                    return ExecutionCheck(
                        can_proceed=False,
                        reason=f"Circuit breaker active: {self.circuit_breaker.trigger_reason}",
                        metadata={'circuit_breaker_triggered': True}
                    )
                else:
                    # Cooldown expired, reset
                    self.circuit_breaker.triggered = False
                    self.circuit_breaker.trigger_time = None
                    self.circuit_breaker.cooldown_until = None
                    self.status = SupervisorStatus.ACTIVE
                    tprint_info(f"✅ Circuit breaker cooldown expired, execution allowed for {symbol}")
        
        # Check total exposure limit (portfolio-level)
        tprint_info(f"📊 Checking exposure limit: {current_exposure:.2%} vs {self.max_total_exposure:.2%}")
        if current_exposure > self.max_total_exposure:
            reason_msg = f"Total exposure {current_exposure:.2%} exceeds limit {self.max_total_exposure:.2%}"
            reasons.append(reason_msg)
            suggested_adjustments['reduce_position_size'] = True
            suggested_adjustments['target_exposure'] = self.max_total_exposure * 0.9
            tprint_warning(f"⚠️ Exposure limit exceeded for {symbol}: {reason_msg}")
        else:
            tprint_info(f"✓ Exposure limit check passed for {symbol}")
        
        # Check execution quality (if we have recent poor execution stats)
        if self.track_execution_metrics:
            tprint_info(f"📈 Checking execution quality trends for {symbol}...")
            quality_check = self._check_execution_quality_trends()
            if not quality_check['acceptable']:
                reason_msg = f"Execution quality below threshold: {quality_check['reason']}"
                reasons.append(reason_msg)
                if quality_check.get('suggest_reduce_size'):
                    suggested_adjustments['reduce_position_size'] = True
                tprint_warning(f"⚠️ Execution quality check failed for {symbol}: {reason_msg}")
            else:
                tprint_info(f"✓ Execution quality check passed for {symbol}")
        
        # Check recent rejection rate
        rejection_count: int = len(self.recent_rejections)
        if rejection_count >= self.max_rejections_per_minute:
            tprint_error(f"🚨 Too many rejections ({rejection_count}) in last minute, blocking execution for {symbol}")
            return ExecutionCheck(
                can_proceed=False,
                reason=f"Too many rejections in last minute: {rejection_count}",
                metadata={'rejection_count': rejection_count}
            )
        elif rejection_count > 0:
            tprint_info(f"⚠️ {rejection_count} rejections in last minute for {symbol} (limit: {self.max_rejections_per_minute})")
        
        can_proceed: bool = len(reasons) == 0
        
        if can_proceed:
            tprint_success(f"✅ Pre-execution check passed for {symbol} ({action})")
        else:
            tprint_error(f"❌ Pre-execution check failed for {symbol}: {'; '.join(reasons)}")
        
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
        tprint_info(f"📊 Monitoring execution for order {order_id}...")
        try:
            # Thread-safe execution stats update
            async with self._execution_stats_lock:
                self.execution_stats['total_orders'] += 1
                total_orders: int = self.execution_stats['total_orders']
                
                status: str = execution_result.get('status', '').upper()
                tprint_info(f"📋 Order {order_id} status: {status}")
                
                if status in ['FILLED', 'PARTIALLY_FILLED']:
                    self.execution_stats['filled_orders'] += 1
                    filled_orders: int = self.execution_stats['filled_orders']
                    fill_rate: float = filled_orders / total_orders if total_orders > 0 else 0.0
                    tprint_success(f"✅ Order {order_id} filled (fill rate: {fill_rate:.2%})")
                    
                    # Track slippage if available
                    if 'slippage' in execution_result:
                        slippage: float = abs(execution_result['slippage'])
                        tprint_info(f"📈 Order {order_id} slippage: {slippage:.4%}")
                        
                        # Check for excessive slippage
                        if slippage > self.max_slippage_per_trade:
                            tprint_warning(
                                f"⚠️ High slippage detected: {slippage:.4%} for order {order_id} (limit: {self.max_slippage_per_trade:.4%})"
                            )
                        
                        # Update recent executions (keep this list for averaging)
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
                    
                    rejection_count: int = len(self.recent_rejections)
                    tprint_warning(f"⚠️ Order {order_id} {status.lower()} (rejection count: {rejection_count})")
                    
                    # Clean old rejections (older than 1 minute)
                    cutoff: datetime = datetime.now() - timedelta(minutes=1)
                    self.recent_rejections = [r for r in self.recent_rejections if r > cutoff]
            
            # Check circuit breaker trigger (outside lock to avoid deadlock)
            rejection_count = len(self.recent_rejections)
            if rejection_count >= self.max_rejections_per_minute:
                tprint_error(f"🚨 Rejection limit reached ({rejection_count}), triggering circuit breaker")
                await self._trigger_circuit_breaker(
                    f"Too many rejections: {rejection_count} in last minute"
                )
            
            # Track commission if available (thread-safe)
            async with self._execution_stats_lock:
                if 'commission' in execution_result:
                    commission: float = execution_result['commission']
                    self.execution_stats['total_commissions'] += commission
                    tprint_info(f"💰 Order {order_id} commission: {commission:.4f}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error monitoring execution: {e}")
            tprint_error(f"❌ Error monitoring execution for order {order_id}: {e}")

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
        tprint_info(f"📊 Post-trade analysis for trade {trade_id}...")
        try:
            # Extract PnL if available
            pnl: float = trade_outcome.get('pnl_absolute', 0.0)
            tprint_info(f"💰 Trade {trade_id} PnL: {pnl:.4f}")
            
            if pnl < 0:  # Loss
                loss_amount: float = abs(pnl)
                now: datetime = datetime.now()
                
                tprint_warning(f"⚠️ Loss detected for trade {trade_id}: {loss_amount:.4f}")
                
                # Track hourly losses
                self.hourly_losses.append((now, loss_amount))
                cutoff: datetime = now - timedelta(hours=1)
                self.hourly_losses = [(t, l) for t, l in self.hourly_losses if t > cutoff]
                hourly_loss_total: float = sum(l for _, l in self.hourly_losses)
                tprint_info(f"📈 Hourly loss total: {hourly_loss_total:.4f} ({len(self.hourly_losses)} losses)")
                
                # Track daily losses
                self.daily_losses.append((now, loss_amount))
                daily_cutoff: datetime = now - timedelta(days=1)
                self.daily_losses = [(t, l) for t, l in self.daily_losses if t > daily_cutoff]
                daily_loss_total: float = sum(l for _, l in self.daily_losses)
                tprint_info(f"📅 Daily loss total: {daily_loss_total:.4f} ({len(self.daily_losses)} losses)")
                
                # Check circuit breakers
                if self.circuit_breakers_enabled:
                    tprint_info(f"🛡️ Checking loss-based circuit breakers for trade {trade_id}...")
                    await self._check_loss_based_circuit_breakers()
            else:
                tprint_success(f"✅ Profit for trade {trade_id}: {pnl:.4f}")
            
            # Update execution quality metrics
            if 'slippage' in trade_outcome:
                tprint_info(f"📈 Updating execution metrics for trade {trade_id}...")
                await self.monitor_execution(
                    trade_id,
                    {'status': 'FILLED', 'slippage': trade_outcome['slippage']}
                )
            
            tprint_success(f"✅ Post-trade analysis completed for trade {trade_id}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error in post-trade analysis: {e}")
            tprint_error(f"❌ Error in post-trade analysis for trade {trade_id}: {e}")

    @handles_errors
    async def _check_portfolio_risk_limits(
        self,
        symbol: str,
        current_positions: Dict[str, Dict[str, Any]],
        account_balance: Optional[float]
    ) -> Tuple[bool, List[str]]:
        """
        Check portfolio-level risk limits using RiskCalculator.
        
        Uses proper risk calculation that accounts for volatility and stop-loss distances,
        not just simple exposure.
        """
        if not account_balance or account_balance <= 0:
            return True, []  # Skip check if balance unavailable
        
        # Update account balance tracking
        self.account_balance = account_balance
        
        reasons: List[str] = []
        
        # Calculate total portfolio risk using RiskCalculator for each position
        total_risk: float = 0.0
        for pos_id, position in current_positions.items():
            quantity: float = position.get('quantity', 0)
            entry_price: float = position.get('entry_price', 0)
            stop_loss_price: Optional[float] = position.get('stop_loss_price')
            volatility: Optional[float] = position.get('volatility')  # If available
            
            if quantity > 0 and entry_price > 0:
                # Use RiskCalculator for proper risk calculation
                try:
                    risk_metrics = await self.risk_calculator.calculate_risk_metrics(
                        position_size=quantity * entry_price,
                        current_price=entry_price,
                        account_balance=account_balance,
                        volatility=volatility,
                        stop_loss_price=stop_loss_price
                    )
                    # Use position_risk from metrics (which accounts for stop-loss)
                    total_risk += risk_metrics.position_risk
                except Exception as e:
                    # Fallback to simple calculation if RiskCalculator fails
                    self.logger.warning(f"RiskCalculator failed for position {pos_id}: {e}")
                    tprint_warning(f"⚠️ RiskCalculator failed for position {pos_id}: {e}, using fallback")
                    pos_value: float = quantity * entry_price
                    total_risk += pos_value / account_balance
        
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
        
        reasons: List[str] = []
        
        # Calculate total exposure
        total_exposure: float = 0.0
        for pos_id, position in current_positions.items():
            pos_value: float = position.get('quantity', 0) * position.get('entry_price', 0)
            leverage: float = position.get('leverage', 1.0)
            exposure: float = (pos_value * leverage) / account_balance
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
        decision: "TradingDecision",
        current_positions: Optional[Dict[str, Dict[str, Any]]],
        account_balance: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Check cross-asset exposure limits to avoid over-correlation.
        
        This checks that we don't have too much exposure in correlated asset groups.
        Single-asset limits are handled elsewhere.
        
        Note: Recalculates exposure from actual positions instead of accumulating.
        """
        if not account_balance or account_balance <= 0:
            return True, ""  # Skip check if balance unavailable
        
        # Update account balance tracking
        self.account_balance = account_balance
        
        if not current_positions:
            current_positions = {}
        
        # Get decision details
        symbol = getattr(decision, 'symbol', '')
        decision_action = getattr(decision, 'action', '').lower()
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
        
        # Recalculate current exposure for this asset group from actual positions
        current_group_exposure = 0.0
        
        for pos_id, position in current_positions.items():
            pos_symbol = position.get('symbol', '')
            if pos_symbol in self.correlated_asset_groups.get(symbol_group, []):
                pos_value = position.get('quantity', 0) * position.get('entry_price', 0)
                leverage = position.get('leverage', 1.0)
                exposure = (pos_value * leverage) / account_balance
                current_group_exposure += exposure
        
        # Calculate new exposure based on decision action
        if decision_action in ['buy', 'open']:
            # Adding new position
            decision_exposure = decision_value / account_balance
            new_group_exposure = current_group_exposure + decision_exposure
        elif decision_action in ['sell', 'close']:
            # Closing/reducing position - subtract exposure
            decision_exposure = decision_value / account_balance
            new_group_exposure = max(0.0, current_group_exposure - decision_exposure)
        else:
            # Hold or unknown action - no change
            new_group_exposure = current_group_exposure
        
        # Check against limit
        if new_group_exposure > self.max_cross_asset_exposure:
            return False, (
                f"Cross-asset exposure for group '{symbol_group}' would be {new_group_exposure:.2%}, "
                f"exceeding limit {self.max_cross_asset_exposure:.2%}. "
                f"Current exposure: {current_group_exposure:.2%}"
            )
        
        # Update tracking (recalculate all groups)
        await self._recalculate_cross_asset_exposure(current_positions, account_balance, decision)
        
        return True, ""

    @handles_errors
    async def _check_portfolio_risk_with_decision(
        self,
        decision: "TradingDecision",
        current_positions: Optional[Dict[str, Dict[str, Any]]],
        account_balance: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Check portfolio risk including the new decision.
        
        Properly handles position openings, closings, and modifications.
        """
        if not account_balance or account_balance <= 0:
            return True, ""  # Skip check if balance unavailable
        
        # Update account balance tracking
        self.account_balance = account_balance
        
        if not current_positions:
            current_positions = {}
        
        # Get decision details
        decision_action = getattr(decision, 'action', '').lower()
        decision_symbol = getattr(decision, 'symbol', '')
        decision_quantity = getattr(decision, 'quantity', 0)
        decision_price = getattr(decision, 'price', 0)
        decision_value = decision_quantity * decision_price
        
        # Calculate risk from decision
        decision_risk = 0.0
        if decision_action in ['buy', 'open']:
            # Adding new position - calculate risk
            stop_loss_price = getattr(decision, 'stop_loss_price', None)
            volatility = getattr(decision, 'volatility', None)
            
            try:
                risk_metrics = await self.risk_calculator.calculate_risk_metrics(
                    position_size=decision_value,
                    current_price=decision_price,
                    account_balance=account_balance,
                    volatility=volatility,
                    stop_loss_price=stop_loss_price
                )
                decision_risk = risk_metrics.position_risk
            except Exception as e:
                # Fallback to simple calculation
                self.logger.warning(f"RiskCalculator failed for decision: {e}")
                decision_risk = decision_value / account_balance
        
        # Calculate current portfolio risk (excluding the symbol being modified)
        current_risk = 0.0
        for pos_id, position in current_positions.items():
            pos_symbol = position.get('symbol', '')
            # Skip risk from position being closed or modified
            if decision_action in ['sell', 'close'] and pos_symbol == decision_symbol:
                continue
            
            quantity = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            stop_loss_price = position.get('stop_loss_price')
            volatility = position.get('volatility')
            
            if quantity > 0 and entry_price > 0:
                try:
                    risk_metrics = await self.risk_calculator.calculate_risk_metrics(
                        position_size=quantity * entry_price,
                        current_price=entry_price,
                        account_balance=account_balance,
                        volatility=volatility,
                        stop_loss_price=stop_loss_price
                    )
                    current_risk += risk_metrics.position_risk
                except Exception as e:
                    # Fallback
                    pos_value = quantity * entry_price
                    current_risk += pos_value / account_balance
        
        # Calculate new total risk
        new_total_risk = current_risk + decision_risk
        
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
        warnings: List[str] = []
        
        if self.monitor_data_quality:
            # Check data freshness if timestamp available
            if 'timestamp' in market_snapshot:
                snapshot_timestamp: datetime = market_snapshot['timestamp']
                data_age: float = (datetime.now() - snapshot_timestamp).total_seconds()
                if data_age > 300:  # 5 minutes
                    warning_msg: str = f"Market data is {data_age:.0f}s old"
                    warnings.append(warning_msg)
                    tprint_warning(f"⚠️ System health warning: {warning_msg}")
        
        # Add more health checks as needed
        
        return len(warnings) == 0, warnings

    def _check_execution_quality_trends(self) -> Dict[str, Any]:
        """Check execution quality trends."""
        stats: Dict[str, Any] = self.execution_stats
        
        if stats['total_orders'] == 0:
            return {'acceptable': True, 'reason': 'No execution history'}
        
        # Calculate fill rate
        fill_rate: float = stats['filled_orders'] / stats['total_orders']
        tprint_info(f"📊 Execution quality: fill_rate={fill_rate:.2%}, threshold={self.min_fill_rate:.2%}")
        
        if fill_rate < self.min_fill_rate:
            return {
                'acceptable': False,
                'reason': f"Fill rate {fill_rate:.2%} below threshold {self.min_fill_rate:.2%}",
                'suggest_reduce_size': True
            }
        
        # Calculate average slippage from recent executions only
        if len(stats['recent_executions']) > 0:
            slippage_sum: float = sum(ex.get('slippage', 0) for ex in stats['recent_executions'])
            avg_slippage: float = slippage_sum / len(stats['recent_executions'])
            tprint_info(f"📊 Average slippage: {avg_slippage:.4%}, threshold={self.max_avg_slippage:.4%}")
            if avg_slippage > self.max_avg_slippage:
                return {
                    'acceptable': False,
                    'reason': f"Average slippage {avg_slippage:.4%} exceeds threshold {self.max_avg_slippage:.4%}",
                    'suggest_reduce_size': True
                }
        
        return {'acceptable': True, 'reason': 'Execution quality acceptable'}

    async def _check_loss_based_circuit_breakers(self) -> None:
        """
        Check if loss-based circuit breakers should trigger.
        
        Checks hourly and daily loss limits against account balance.
        Triggers circuit breaker if thresholds exceeded.
        """
        if not self.circuit_breakers_enabled:
            return
        
        # Need account balance to calculate percentage losses
        account_balance: Optional[float] = self.account_balance
        if not account_balance or account_balance <= 0:
            self.logger.warning("⚠️ Cannot check loss-based circuit breakers: account balance unavailable")
            tprint_warning("⚠️ Cannot check loss-based circuit breakers: account balance unavailable")
            return
        
        # Calculate hourly loss percentage
        if self.hourly_losses:
            hourly_loss_total: float = sum(loss for _, loss in self.hourly_losses)
            hourly_loss_pct: float = hourly_loss_total / account_balance
            tprint_info(f"📊 Hourly loss: {hourly_loss_pct:.2%} (limit: {self.max_loss_per_hour:.2%})")
            
            if hourly_loss_pct > self.max_loss_per_hour:
                tprint_error(f"🚨 Hourly loss limit exceeded: {hourly_loss_pct:.2%} > {self.max_loss_per_hour:.2%}")
                await self._trigger_circuit_breaker(
                    f"Hourly loss {hourly_loss_pct:.2%} exceeds limit {self.max_loss_per_hour:.2%}"
                )
                return  # Don't check daily if hourly already triggered
        
        # Calculate daily loss percentage
        if self.daily_losses:
            daily_loss_total: float = sum(loss for _, loss in self.daily_losses)
            daily_loss_pct: float = daily_loss_total / account_balance
            tprint_info(f"📊 Daily loss: {daily_loss_pct:.2%} (limit: {self.max_loss_per_day:.2%})")
            
            if daily_loss_pct > self.max_loss_per_day:
                tprint_error(f"🚨 Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_loss_per_day:.2%}")
                await self._trigger_circuit_breaker(
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_loss_per_day:.2%}"
                )

    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker with thread safety."""
        if not self.circuit_breakers_enabled:
            return
        
        # Thread-safe circuit breaker trigger
        async with self._circuit_breaker_lock:
            # Don't re-trigger if already triggered
            if self.circuit_breaker.triggered:
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

    async def _recalculate_cross_asset_exposure(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        account_balance: float,
        decision: Optional["TradingDecision"] = None
    ) -> None:
        """
        Recalculate cross-asset exposure for all groups from actual positions.
        
        This ensures exposure tracking is accurate when positions change.
        """
        # Clear existing exposure tracking
        self.cross_asset_exposure.clear()
        
        # Calculate exposure per asset group
        for pos_id, position in current_positions.items():
            pos_symbol = position.get('symbol', '')
            pos_value = position.get('quantity', 0) * position.get('entry_price', 0)
            leverage = position.get('leverage', 1.0)
            exposure = (pos_value * leverage) / account_balance
            
            # Find which asset group this symbol belongs to
            for group_name, symbols in self.correlated_asset_groups.items():
                if pos_symbol in symbols:
                    if group_name not in self.cross_asset_exposure:
                        self.cross_asset_exposure[group_name] = 0.0
                    self.cross_asset_exposure[group_name] += exposure
                    break
        
        # If decision is adding a position, include it
        if decision:
            decision_action = getattr(decision, 'action', '').lower()
            if decision_action in ['buy', 'open']:
                decision_symbol = getattr(decision, 'symbol', '')
                decision_quantity = getattr(decision, 'quantity', 0)
                decision_price = getattr(decision, 'price', 0)
                decision_value = decision_quantity * decision_price
                decision_exposure = decision_value / account_balance
                
                # Find which asset group this symbol belongs to
                for group_name, symbols in self.correlated_asset_groups.items():
                    if decision_symbol in symbols:
                        if group_name not in self.cross_asset_exposure:
                            self.cross_asset_exposure[group_name] = 0.0
                        self.cross_asset_exposure[group_name] += decision_exposure
                        break

    def _calculate_cross_asset_exposure(self) -> Dict[str, float]:
        """Calculate current cross-asset exposure per group."""
        return self.cross_asset_exposure.copy()

    async def update_account_balance(self, account_balance: float) -> None:
        """
        Update account balance tracking.
        
        Args:
            account_balance: Current account balance
        """
        if account_balance > 0:
            old_balance: Optional[float] = self.account_balance
            self.account_balance = account_balance
            if old_balance is not None:
                change: float = account_balance - old_balance
                change_pct: float = (change / old_balance) * 100 if old_balance > 0 else 0.0
                tprint_info(f"💰 Account balance updated: {account_balance:.2f} (change: {change:+.2f}, {change_pct:+.2f}%)")
            else:
                tprint_info(f"💰 Account balance set: {account_balance:.2f}")
            self.logger.debug(f"Account balance updated: {account_balance:.2f}")
        else:
            self.logger.warning(f"Invalid account balance provided: {account_balance}")
            tprint_warning(f"⚠️ Invalid account balance provided: {account_balance}")

    async def update_positions(
        self,
        positions_by_symbol: Dict[str, Dict[str, Any]],
        account_balance: float
    ) -> None:
        """
        Update supervisor's view of all active positions across all symbols.
        
        This allows the Supervisor to track portfolio-level exposure.
        Thread-safe version that properly handles position structure.
        
        Args:
            positions_by_symbol: Dict of symbol -> positions dict
            account_balance: Current account balance
        """
        tprint_info(f"🔄 Updating positions for {len(positions_by_symbol)} symbols...")
        
        # Update account balance
        await self.update_account_balance(account_balance)
        
        try:
            # Thread-safe positions update
            async with self._positions_lock:
                self.all_active_positions = positions_by_symbol.copy()
                
                # Recalculate portfolio metrics
                total_exposure: float = 0.0
                total_risk: float = 0.0
                position_count: int = 0
                
                for symbol, positions in positions_by_symbol.items():
                    # Positions dict may contain multiple position entries
                    if isinstance(positions, dict):
                        # Check if it's a single position dict or multiple
                        if 'quantity' in positions:
                            # Single position
                            quantity: float = positions.get('quantity', 0)
                            entry_price: float = positions.get('entry_price', 0)
                            leverage: float = positions.get('leverage', 1.0)
                            
                            if quantity > 0 and entry_price > 0:
                                symbol_value: float = quantity * entry_price
                                symbol_risk: float = symbol_value / account_balance
                                total_exposure += (symbol_value * leverage) / account_balance
                                total_risk += symbol_risk
                                position_count += 1
                                tprint_info(f"📊 Position {symbol}: exposure={(symbol_value * leverage) / account_balance:.2%}, risk={symbol_risk:.2%}")
                        else:
                            # Multiple positions keyed by position_id
                            for pos_id, position in positions.items():
                                if isinstance(position, dict) and 'quantity' in position:
                                    quantity = position.get('quantity', 0)
                                    entry_price = position.get('entry_price', 0)
                                    leverage = position.get('leverage', 1.0)
                                    
                                    if quantity > 0 and entry_price > 0:
                                        pos_value: float = quantity * entry_price
                                        pos_risk: float = pos_value / account_balance
                                        total_exposure += (pos_value * leverage) / account_balance
                                        total_risk += pos_risk
                                        position_count += 1
                
                self.total_portfolio_exposure = total_exposure
                self.total_portfolio_risk = total_risk
                
                tprint_structured(
                    "Portfolio Metrics Updated",
                    {
                        "total_positions": position_count,
                        "total_exposure": f"{total_exposure:.2%}",
                        "total_risk": f"{total_risk:.2%}",
                        "symbols": list(positions_by_symbol.keys())
                    },
                    level=LogLevel.INFO
                )
                
                # Recalculate cross-asset exposure from actual positions
                await self._recalculate_cross_asset_exposure(
                    self._flatten_positions(positions_by_symbol),
                    account_balance
                )
            
            tprint_success(f"✅ Positions updated: {position_count} positions across {len(positions_by_symbol)} symbols")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating positions: {e}")
            tprint_error(f"❌ Error updating positions: {e}")

    def _flatten_positions(
        self,
        positions_by_symbol: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Flatten positions_by_symbol into a single dict keyed by position_id.
        
        Args:
            positions_by_symbol: Dict of symbol -> positions dict
            
        Returns:
            Flattened dict of position_id -> position
        """
        flattened: Dict[str, Dict[str, Any]] = {}
        for symbol, positions in positions_by_symbol.items():
            if isinstance(positions, dict):
                if 'quantity' in positions:
                    # Single position - use symbol as key
                    flattened[f"{symbol}_0"] = {**positions, 'symbol': symbol}
                else:
                    # Multiple positions
                    for pos_id, position in positions.items():
                        if isinstance(position, dict) and 'quantity' in position:
                            flattened[pos_id] = {**position, 'symbol': symbol}
        return flattened

    async def remove_position(
        self,
        symbol: str,
        position_id: Optional[str] = None,
        account_balance: Optional[float] = None
    ) -> None:
        """
        Remove a position from tracking.
        
        Args:
            symbol: Trading symbol
            position_id: Optional position ID (if None, removes all positions for symbol)
            account_balance: Current account balance (for recalculation)
        """
        tprint_info(f"🗑️ Removing position for {symbol}" + (f" (position_id: {position_id})" if position_id else " (all positions)"))
        
        async with self._positions_lock:
            if symbol in self.all_active_positions:
                positions: Dict[str, Any] = self.all_active_positions[symbol]
                
                if position_id and isinstance(positions, dict) and 'quantity' not in positions:
                    # Multiple positions - remove specific one
                    if position_id in positions:
                        del positions[position_id]
                        if not positions:
                            del self.all_active_positions[symbol]
                            tprint_info(f"✓ Removed all positions for {symbol}")
                        else:
                            tprint_info(f"✓ Removed position {position_id} for {symbol}")
                    else:
                        tprint_warning(f"⚠️ Position {position_id} not found for {symbol}")
                else:
                    # Single position or remove all - remove entire symbol entry
                    del self.all_active_positions[symbol]
                    tprint_info(f"✓ Removed all positions for {symbol}")
            else:
                tprint_warning(f"⚠️ No positions found for {symbol}")
            
            # Recalculate metrics if balance provided
            if account_balance:
                tprint_info(f"📊 Recalculating portfolio metrics after position removal for {symbol}...")
                await self.update_account_balance(account_balance)
                await self._recalculate_cross_asset_exposure(
                    self._flatten_positions(self.all_active_positions),
                    account_balance
                )
                tprint_success(f"✅ Portfolio metrics updated after removing position for {symbol}")

    def get_supervisor_status(self) -> Dict[str, Any]:
        """Get current supervisor status and metrics."""
        fill_rate: float = (
            self.execution_stats['filled_orders'] / self.execution_stats['total_orders']
            if self.execution_stats['total_orders'] > 0 else 0.0
        )
        avg_slippage: float = (
            sum(ex.get('slippage', 0) for ex in self.execution_stats['recent_executions']) / len(self.execution_stats['recent_executions'])
            if len(self.execution_stats['recent_executions']) > 0 else 0.0
        )
        
        status_dict: Dict[str, Any] = {
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
                'fill_rate': fill_rate,
                'avg_slippage': avg_slippage
            },
            'recent_rejections_count': len(self.recent_rejections)
        }
        
        tprint_structured(
            "Supervisor Status",
            {
                "status": status_dict['status'],
                "portfolio_risk": f"{status_dict['portfolio_metrics']['total_portfolio_risk']:.2%}",
                "portfolio_exposure": f"{status_dict['portfolio_metrics']['total_portfolio_exposure']:.2%}",
                "fill_rate": f"{fill_rate:.2%}",
                "circuit_breaker_triggered": status_dict['circuit_breaker']['triggered']
            },
            level=LogLevel.INFO
        )
        
        return status_dict

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
    """
    Create a Trading Supervisor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TradingSupervisor: Initialized Trading Supervisor instance
    """
    tprint_info("🏭 Creating Trading Supervisor instance...")
    supervisor: TradingSupervisor = TradingSupervisor(config)
    tprint_success("✅ Trading Supervisor instance created")
    return supervisor
