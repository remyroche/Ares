"""
Comprehensive Trade Monitor

Advanced monitoring system for trading operations with detailed metrics,
ML model explanations, SHAP/LIME analysis, and comprehensive reporting.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel
# Import from trading module components
# from ..alert_manager import AlertManager, AlertType, AlertPriority  # Removed to avoid circular imports
# from ..regime_monitor import RegimeMonitor, RegimeState, RegimeType  # Removed to avoid circular imports

# Performance tracker imports would be added here if needed
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.validation import validate_trading_config
from ..utils.helpers import format_trading_metrics, save_trading_data

logger = system_logger.getChild('ComprehensiveTradeMonitor')

@dataclass
class DetailedTradeMetrics:
    """Comprehensive metrics for a single trade."""
    # Basic trade information
    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # buy, sell, hold, close
    quantity: float
    price: float
    
    # ML Model Information
    models_used: Dict[str, Any] = field(default_factory=dict)  # model_id -> model_info
    model_predictions: Dict[str, float] = field(default_factory=dict)  # model_id -> prediction
    model_confidences: Dict[str, float] = field(default_factory=dict)  # model_id -> confidence
    model_weights: Dict[str, float] = field(default_factory=dict)  # model_id -> ensemble_weight
    
    # Signal Information
    analyst_signal: Optional[Dict[str, Any]] = None
    tactician_signal: Optional[Dict[str, Any]] = None
    combined_signal: Optional[Dict[str, Any]] = None
    signal_confidence: float = 0.0
    signal_strength: float = 0.0
    
    # Regime Information
    regime_type: str = "unknown"
    regime_confidence: float = 0.0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    regime_stability: float = 0.0
    
    # Position Sizing
    position_size: float = 0.0
    leverage: float = 1.0
    kelly_fraction: float = 0.0
    risk_per_trade: float = 0.02
    
    # Risk Metrics
    portfolio_risk: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown_risk: float = 0.0
    volatility_estimate: float = 0.0
    
    # SHAP/LIME Explanations
    shap_explanations: Dict[str, Dict[str, float]] = field(default_factory=dict)  # model_id -> feature_explanations
    lime_explanations: Dict[str, Dict[str, float]] = field(default_factory=dict)  # model_id -> feature_explanations
    feature_importance: Dict[str, float] = field(default_factory=dict)  # feature -> importance
    
    # Market Context
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    support_resistance_levels: Dict[str, float] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Performance Metrics (filled after trade completion)
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_absolute: Optional[float] = None
    pnl_percentage: Optional[float] = None
    duration_minutes: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    
    # Trade Quality Metrics
    execution_quality: float = 0.0  # How well the trade was executed
    slippage: float = 0.0
    commission: float = 0.0
    timing_quality: float = 0.0  # How good was the entry/exit timing
    
    # Metadata
    trading_mode: str = "paper"
    exchange: str = "binance"
    strategy_version: str = "1.0"
    model_versions: Dict[str, str] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetailedTradeMetrics':
        """Create from dictionary."""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class TradingSessionMetrics:
    """Comprehensive metrics for a trading session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # Performance Metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    
    # Model Performance
    model_accuracy: Dict[str, float] = field(default_factory=dict)
    model_usage_frequency: Dict[str, int] = field(default_factory=dict)
    model_contribution_to_pnl: Dict[str, float] = field(default_factory=dict)
    
    # Regime Performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_frequency: Dict[str, int] = field(default_factory=dict)
    
    # Execution Quality
    avg_slippage: float = 0.0
    avg_execution_time_ms: float = 0.0
    execution_success_rate: float = 0.0
    
    # Detailed Trade List
    trades: List[DetailedTradeMetrics] = field(default_factory=list)

class ComprehensiveTradeMonitor:
    """
    Comprehensive trade monitoring system with detailed metrics,
    ML explanations, and advanced reporting capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger.getChild('ComprehensiveTradeMonitor')
        
        # Core monitoring components
        self.enhanced_monitoring = EnhancedMonitoringOrchestrator()
        self.explainability_integrator = ExplainabilityIntegrator()
        self.explainability_orchestrator = ExplainabilityOrchestrator()
        
        # Trade storage
        self.active_trades: Dict[str, DetailedTradeMetrics] = {}
        self.completed_trades: List[DetailedTradeMetrics] = []
        self.current_session: Optional[TradingSessionMetrics] = None
        
        # Configuration
        self.max_trades_in_memory = self.config.get('max_trades_in_memory', 10000)
        self.enable_explanations = self.config.get('enable_explanations', True)
        self.enable_real_time_export = self.config.get('enable_real_time_export', True)
        self.export_directory = Path(self.config.get('export_directory', 'trading_reports'))
        
        # Performance tracking
        self.performance_cache: Dict[str, Any] = {}
        self.last_performance_update = datetime.now()
        
        # Initialize monitoring components
        self.is_initialized = False
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def initialize(self) -> bool:
        """Initialize the comprehensive trade monitor."""
        try:
            tprint_info("🚀 Initializing Comprehensive Trade Monitor...")
            
            # Initialize enhanced monitoring orchestrator
            monitoring_config = EnhancedMonitoringConfig(
                enable_monitoring=True,
                enable_explanations=self.enable_explanations,
                enable_real_time_tracking=True,
                monthly_export_enabled=True,
                daily_export_enabled=True,
                export_directory=str(self.export_directory),
                enable_shap=True,
                enable_lime=True,
                auto_integrate_trading_systems=True
            )
            
            success = await self.enhanced_monitoring.initialize(monitoring_config)
            if not success:
                raise TradingError("Failed to initialize enhanced monitoring")
            
            tprint_success("✅ Enhanced monitoring orchestrator initialized")
            
            # Initialize explainability components
            await self.explainability_integrator.initialize()
            await self.explainability_orchestrator.initialize()
            
            tprint_success("✅ Explainability components initialized")
            
            # Create export directory
            self.export_directory.mkdir(parents=True, exist_ok=True)
            
            # Start new trading session
            await self.start_new_session()
            
            self.is_initialized = True
            tprint_success("✅ Comprehensive Trade Monitor initialized successfully")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Comprehensive Trade Monitor: {e}")
            raise
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def start_new_session(self) -> str:
        """Start a new trading session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.current_session = TradingSessionMetrics(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        tprint_info(f"📊 Started new trading session: {session_id}")
        return session_id
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=False
    )
    async def record_trade_decision(
        self,
        trade_data: Dict[str, Any],
        models_used: Optional[Dict[str, Any]] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Record a comprehensive trade decision with all metrics.
        
        Args:
            trade_data: Basic trade information
            models_used: Dictionary of ML models used
            market_data: Market data context
            
        Returns:
            Trade ID for tracking
        """
        try:
            if not self.is_initialized:
                tprint_warning("⚠️ Trade monitor not initialized, initializing now...")
                await self.initialize()
            
            # Generate unique trade ID
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            tprint_info(f"📝 Recording trade decision: {trade_id}")
            
            # Create detailed trade metrics
            trade_metrics = DetailedTradeMetrics(
                trade_id=trade_id,
                timestamp=datetime.now(),
                symbol=trade_data.get('symbol', 'UNKNOWN'),
                action=trade_data.get('action', 'unknown'),
                quantity=trade_data.get('quantity', 0.0),
                price=trade_data.get('price', 0.0),
                trading_mode=trade_data.get('trading_mode', 'paper'),
                exchange=trade_data.get('exchange', 'binance')
            )
            
            # Extract ML model information
            if models_used:
                await self._extract_model_information(trade_metrics, models_used, market_data)
            
            # Extract signal information
            if 'analyst_signal' in trade_data:
                trade_metrics.analyst_signal = trade_data['analyst_signal']
            if 'tactician_signal' in trade_data:
                trade_metrics.tactician_signal = trade_data['tactician_signal']
            if 'combined_signal' in trade_data:
                trade_metrics.combined_signal = trade_data['combined_signal']
            
            # Extract regime information
            if 'regime_data' in trade_data:
                await self._extract_regime_information(trade_metrics, trade_data['regime_data'])
            
            # Extract position sizing information
            if 'position_sizing' in trade_data:
                await self._extract_position_sizing_information(trade_metrics, trade_data['position_sizing'])
            
            # Extract risk metrics
            if 'risk_metrics' in trade_data:
                await self._extract_risk_metrics(trade_metrics, trade_data['risk_metrics'])
            
            # Extract market context
            if market_data is not None:
                await self._extract_market_context(trade_metrics, market_data)
            
            # Generate explanations if enabled
            if self.enable_explanations and models_used:
                await self._generate_explanations(trade_metrics, models_used, market_data)
            
            # Store trade
            self.active_trades[trade_id] = trade_metrics
            
            # Update session metrics
            if self.current_session:
                self.current_session.total_trades += 1
                self.current_session.trades.append(trade_metrics)
            
            # Record in enhanced monitoring system
            await self._record_in_enhanced_monitoring(trade_metrics)
            
            # Export in real-time if enabled
            if self.enable_real_time_export:
                await self._export_trade_metrics(trade_metrics)
            
            tprint_success(f"✅ Trade decision recorded: {trade_id}")
            
            return trade_id
            
        except Exception as e:
            tprint_error(f"❌ Failed to record trade decision: {e}")
            return ""
    
    async def _extract_model_information(
        self,
        trade_metrics: DetailedTradeMetrics,
        models_used: Dict[str, Any],
        market_data: Optional[pd.DataFrame]
    ):
        """Extract detailed ML model information."""
        try:
            for model_id, model_info in models_used.items():
                # Store model metadata
                trade_metrics.models_used[model_id] = {
                    'model_type': model_info.get('type', 'unknown'),
                    'model_version': model_info.get('version', '1.0'),
                    'training_date': model_info.get('training_date'),
                    'features_count': model_info.get('features_count', 0),
                    'model_params': model_info.get('params', {})
                }
                
                # Extract predictions and confidences
                if 'prediction' in model_info:
                    trade_metrics.model_predictions[model_id] = model_info['prediction']
                
                if 'confidence' in model_info:
                    trade_metrics.model_confidences[model_id] = model_info['confidence']
                
                if 'weight' in model_info:
                    trade_metrics.model_weights[model_id] = model_info['weight']
                
                if 'version' in model_info:
                    trade_metrics.model_versions[model_id] = model_info['version']
            
            tprint_info(f"📊 Extracted information for {len(models_used)} models")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract model information: {e}")
    
    async def _extract_regime_information(
        self,
        trade_metrics: DetailedTradeMetrics,
        regime_data: Dict[str, Any]
    ):
        """Extract regime detection information."""
        try:
            trade_metrics.regime_type = regime_data.get('primary_regime', 'unknown')
            trade_metrics.regime_confidence = regime_data.get('confidence', 0.0)
            trade_metrics.regime_probabilities = regime_data.get('regime_probabilities', {})
            trade_metrics.regime_stability = regime_data.get('stability_score', 0.0)
            
            tprint_info(f"🎯 Extracted regime information: {trade_metrics.regime_type}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract regime information: {e}")
    
    async def _extract_position_sizing_information(
        self,
        trade_metrics: DetailedTradeMetrics,
        position_sizing: Dict[str, Any]
    ):
        """Extract position sizing information."""
        try:
            trade_metrics.position_size = position_sizing.get('recommended_size', 0.0)
            trade_metrics.leverage = position_sizing.get('leverage', 1.0)
            trade_metrics.kelly_fraction = position_sizing.get('kelly_fraction', 0.0)
            trade_metrics.risk_per_trade = position_sizing.get('risk_per_trade', 0.02)
            
            tprint_info(f"💰 Extracted position sizing: {trade_metrics.position_size:.4f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract position sizing information: {e}")
    
    async def _extract_risk_metrics(
        self,
        trade_metrics: DetailedTradeMetrics,
        risk_metrics: Dict[str, Any]
    ):
        """Extract risk management metrics."""
        try:
            trade_metrics.portfolio_risk = risk_metrics.get('portfolio_risk', 0.0)
            trade_metrics.var_95 = risk_metrics.get('var_95', 0.0)
            trade_metrics.expected_shortfall = risk_metrics.get('expected_shortfall', 0.0)
            trade_metrics.max_drawdown_risk = risk_metrics.get('max_drawdown_risk', 0.0)
            trade_metrics.volatility_estimate = risk_metrics.get('volatility_estimate', 0.0)
            
            tprint_info(f"⚠️ Extracted risk metrics: VaR95={trade_metrics.var_95:.4f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract risk metrics: {e}")
    
    async def _extract_market_context(
        self,
        trade_metrics: DetailedTradeMetrics,
        market_data: pd.DataFrame
    ):
        """Extract market context from market data."""
        try:
            if market_data.empty:
                return
            
            latest_data = market_data.iloc[-1]
            
            # Basic market conditions
            trade_metrics.market_conditions = {
                'current_price': float(latest_data.get('close', 0)),
                'volume': float(latest_data.get('volume', 0)),
                'volatility': float(market_data['close'].pct_change().rolling(20).std().iloc[-1]) if len(market_data) >= 20 else 0.0,
                'trend_direction': 'up' if len(market_data) >= 2 and latest_data.get('close', 0) > market_data.iloc[-2].get('close', 0) else 'down'
            }
            
            # Technical indicators (if available in data)
            if len(market_data) >= 20:
                trade_metrics.technical_indicators = {
                    'sma_20': float(market_data['close'].rolling(20).mean().iloc[-1]),
                    'rsi': self._calculate_rsi(market_data['close']) if 'close' in market_data.columns else 0.0,
                    'bollinger_position': self._calculate_bollinger_position(market_data['close']) if 'close' in market_data.columns else 0.5
                }
            
            tprint_info("📈 Extracted market context")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract market context: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator using centralized calculator."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            # Import centralized RSI calculator
            from src.feature_generation.indicators import RSICalculator
            
            rsi = RSICalculator.calculate(prices, period)
            return float(rsi.iloc[-1])
            
        except Exception:
            return 50.0
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands."""
        try:
            if len(prices) < period:
                return 0.5
            
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_price = prices.iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]
            
            if upper == lower:
                return 0.5
            
            position = (current_price - lower) / (upper - lower)
            return float(max(0.0, min(1.0, position)))
            
        except Exception:
            return 0.5
    
    async def _generate_explanations(
        self,
        trade_metrics: DetailedTradeMetrics,
        models_used: Dict[str, Any],
        market_data: Optional[pd.DataFrame]
    ):
        """Generate SHAP and LIME explanations for the trade decision."""
        try:
            if market_data is None or market_data.empty:
                tprint_warning("⚠️ No market data available for explanations")
                return
            
            tprint_info("🔍 Generating SHAP/LIME explanations...")
            
            # Prepare features for explanation
            features = self._prepare_features_for_explanation(market_data)
            
            for model_id, model_info in models_used.items():
                try:
                    # Generate SHAP explanation
                    if 'model' in model_info:
                        shap_explanation = await self.explainability_integrator.generate_shap_explanation(
                            model=model_info['model'],
                            model_id=model_id,
                            features=features,
                            feature_names=list(features.keys())
                        )
                        
                        if shap_explanation:
                            trade_metrics.shap_explanations[model_id] = shap_explanation.shap_values
                            tprint_success(f"✅ Generated SHAP explanation for {model_id}")
                    
                    # Generate LIME explanation
                    if 'model' in model_info and hasattr(model_info['model'], 'predict'):
                        lime_explanation = await self.explainability_integrator.generate_lime_explanation(
                            model=model_info['model'],
                            model_id=model_id,
                            features=features,
                            feature_names=list(features.keys())
                        )
                        
                        if lime_explanation:
                            trade_metrics.lime_explanations[model_id] = lime_explanation.explanation
                            tprint_success(f"✅ Generated LIME explanation for {model_id}")
                
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to generate explanations for {model_id}: {e}")
            
            # Calculate overall feature importance
            await self._calculate_overall_feature_importance(trade_metrics)
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate explanations: {e}")
    
    def _prepare_features_for_explanation(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Prepare features for SHAP/LIME explanation."""
        try:
            if market_data.empty:
                return {}
            
            latest_data = market_data.iloc[-1]
            features = {}
            
            # Basic OHLCV features
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in latest_data:
                    features[col] = float(latest_data[col])
            
            # Technical indicators if we have enough data
            if len(market_data) >= 20:
                close_prices = market_data['close']
                
                # Moving averages
                features['sma_5'] = float(close_prices.rolling(5).mean().iloc[-1])
                features['sma_20'] = float(close_prices.rolling(20).mean().iloc[-1])
                
                # Price ratios
                if features.get('sma_20', 0) > 0:
                    features['price_sma20_ratio'] = features.get('close', 0) / features['sma_20']
                
                # Returns
                features['returns_1'] = float(close_prices.pct_change(1).iloc[-1])
                features['returns_5'] = float(close_prices.pct_change(5).iloc[-1])
                
                # Volatility
                features['volatility_20'] = float(close_prices.pct_change().rolling(20).std().iloc[-1])
            
            # Remove any NaN values
            features = {k: v for k, v in features.items() if not pd.isna(v)}
            
            return features
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to prepare features: {e}")
            return {}
    
    async def _calculate_overall_feature_importance(self, trade_metrics: DetailedTradeMetrics):
        """Calculate overall feature importance across all models."""
        try:
            all_features = set()
            
            # Collect all features from SHAP explanations
            for model_id, shap_values in trade_metrics.shap_explanations.items():
                all_features.update(shap_values.keys())
            
            # Calculate weighted average importance
            for feature in all_features:
                total_importance = 0.0
                total_weight = 0.0
                
                for model_id, shap_values in trade_metrics.shap_explanations.items():
                    if feature in shap_values:
                        model_weight = trade_metrics.model_weights.get(model_id, 1.0)
                        importance = abs(shap_values[feature])
                        total_importance += importance * model_weight
                        total_weight += model_weight
                
                if total_weight > 0:
                    trade_metrics.feature_importance[feature] = total_importance / total_weight
            
            tprint_info(f"📊 Calculated importance for {len(trade_metrics.feature_importance)} features")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate feature importance: {e}")
    
    async def _record_in_enhanced_monitoring(self, trade_metrics: DetailedTradeMetrics):
        """Record trade in enhanced monitoring system."""
        try:
            # Convert to enhanced monitoring format
            comprehensive_decision = await self._convert_to_comprehensive_decision(trade_metrics)
            
            # Record in enhanced monitoring
            await self.enhanced_monitoring.record_comprehensive_trade_decision(comprehensive_decision)
            
            tprint_info("📊 Recorded in enhanced monitoring system")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to record in enhanced monitoring: {e}")
    
    async def _convert_to_comprehensive_decision(
        self,
        trade_metrics: DetailedTradeMetrics
    ) -> Dict[str, Any]:
        """Convert trade metrics to enhanced monitoring format."""
        try:
            from src.monitoring.enhanced_ml_monitoring import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
                TradeContext, TradingIndicator, MLModelDecision, EnsembleDecision, TradingMode
            )
            
            # Create trade context
            context = TradeContext(
                exchange=trade_metrics.exchange,
                symbol=trade_metrics.symbol,
                timestamp=trade_metrics.timestamp,
                price=trade_metrics.price,
                volume=trade_metrics.quantity,
                market_conditions=trade_metrics.market_conditions
            )
            
            # Create trading indicators
            indicators = [
                TradingIndicator(
                    name="confidence",
                    value=trade_metrics.signal_confidence,
                    weight=1.0,
                    source="combined_signal"
                ),
                TradingIndicator(
                    name="risk_score",
                    value=trade_metrics.portfolio_risk,
                    weight=1.0,
                    source="risk_calculator"
                )
            ]
            
            # Create individual model decisions
            individual_decisions = []
            for model_id, prediction in trade_metrics.model_predictions.items():
                decision = MLModelDecision(
                    model_id=model_id,
                    model_type=trade_metrics.models_used.get(model_id, {}).get('model_type', 'unknown'),
                    prediction=prediction,
                    confidence=trade_metrics.model_confidences.get(model_id, 0.0),
                    feature_importance=trade_metrics.shap_explanations.get(model_id, {}),
                    execution_time_ms=trade_metrics.execution_time_ms,
                    timestamp=trade_metrics.timestamp
                )
                individual_decisions.append(decision)
            
            # Create ensemble decision
            ensemble_decision = EnsembleDecision(
                final_prediction=trade_metrics.signal_confidence,
                model_weights=trade_metrics.model_weights,
                individual_decisions=individual_decisions,
                ensemble_confidence=trade_metrics.signal_confidence,
                timestamp=trade_metrics.timestamp
            )
            
            # Create comprehensive decision
            comprehensive_decision = {
                'decision_id': trade_metrics.trade_id,
                'timestamp': trade_metrics.timestamp,
                'trading_mode': TradingMode.PAPER if trade_metrics.trading_mode == 'paper' else TradingMode.LIVE,
                'context': context,
                'trading_indicators': indicators,
                'overall_confidence': trade_metrics.signal_confidence,
                'overall_risk_score': trade_metrics.portfolio_risk,
                'ensemble_decision': ensemble_decision,
                'individual_model_decisions': individual_decisions,
                'shap_explanations': trade_metrics.shap_explanations,
                'lime_explanations': trade_metrics.lime_explanations,
                'execution_details': {
                    'action': trade_metrics.action,
                    'quantity': trade_metrics.quantity,
                    'price': trade_metrics.price,
                    'leverage': trade_metrics.leverage
                },
                'performance_metrics': {
                    'pnl_absolute': trade_metrics.pnl_absolute,
                    'pnl_percentage': trade_metrics.pnl_percentage,
                    'execution_quality': trade_metrics.execution_quality
                }
            }
            
            return comprehensive_decision
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to convert to comprehensive decision: {e}")
            return {}
    
    async def _export_trade_metrics(self, trade_metrics: DetailedTradeMetrics):
        """Export trade metrics to file."""
        try:
            # Create daily export directory
            today = datetime.now().strftime('%Y-%m-%d')
            daily_dir = self.export_directory / today
            daily_dir.mkdir(parents=True, exist_ok=True)
            
            # Export individual trade
            trade_file = daily_dir / f"trade_{trade_metrics.trade_id}.json"
            with open(trade_file, 'w') as f:
                json.dump(trade_metrics.to_dict(), f, indent=2, default=str)
            
            # Append to daily summary CSV
            csv_file = daily_dir / f"trades_summary_{today}.csv"
            trade_df = pd.DataFrame([trade_metrics.to_dict()])
            
            if csv_file.exists():
                trade_df.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                trade_df.to_csv(csv_file, index=False)
            
            tprint_info(f"💾 Exported trade metrics to {trade_file}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to export trade metrics: {e}")
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def update_trade_outcome(
        self,
        trade_id: str,
        outcome_data: Dict[str, Any]
    ) -> bool:
        """
        Update trade with outcome data (PnL, exit price, etc.).
        
        Args:
            trade_id: Trade ID to update
            outcome_data: Outcome information
            
        Returns:
            True if update successful
        """
        try:
            if trade_id not in self.active_trades:
                tprint_warning(f"⚠️ Trade {trade_id} not found in active trades")
                return False
            
            trade_metrics = self.active_trades[trade_id]
            
            # Update outcome data
            trade_metrics.exit_price = outcome_data.get('exit_price')
            trade_metrics.pnl_absolute = outcome_data.get('pnl_absolute')
            trade_metrics.pnl_percentage = outcome_data.get('pnl_percentage')
            trade_metrics.duration_minutes = outcome_data.get('duration_minutes')
            trade_metrics.max_favorable_excursion = outcome_data.get('max_favorable_excursion')
            trade_metrics.max_adverse_excursion = outcome_data.get('max_adverse_excursion')
            trade_metrics.execution_quality = outcome_data.get('execution_quality', 0.0)
            trade_metrics.slippage = outcome_data.get('slippage', 0.0)
            trade_metrics.commission = outcome_data.get('commission', 0.0)
            trade_metrics.timing_quality = outcome_data.get('timing_quality', 0.0)
            
            # Move to completed trades
            self.completed_trades.append(trade_metrics)
            del self.active_trades[trade_id]
            
            # Update session metrics
            if self.current_session:
                await self._update_session_metrics(trade_metrics)
            
            # Export updated trade
            if self.enable_real_time_export:
                await self._export_trade_metrics(trade_metrics)
            
            tprint_success(f"✅ Updated trade outcome: {trade_id}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to update trade outcome: {e}")
            return False
    
    async def _update_session_metrics(self, trade_metrics: DetailedTradeMetrics):
        """Update session-level metrics with completed trade."""
        try:
            if not self.current_session:
                return
            
            session = self.current_session
            
            # Update trade counts
            if trade_metrics.pnl_absolute is not None:
                if trade_metrics.pnl_absolute > 0:
                    session.winning_trades += 1
                elif trade_metrics.pnl_absolute < 0:
                    session.losing_trades += 1
                else:
                    session.break_even_trades += 1
                
                # Update PnL metrics
                session.total_pnl += trade_metrics.pnl_absolute
                
                if trade_metrics.pnl_absolute > 0:
                    session.gross_profit += trade_metrics.pnl_absolute
                else:
                    session.gross_loss += abs(trade_metrics.pnl_absolute)
            
            # Update model usage
            for model_id in trade_metrics.models_used.keys():
                session.model_usage_frequency[model_id] = session.model_usage_frequency.get(model_id, 0) + 1
                if trade_metrics.pnl_absolute:
                    session.model_contribution_to_pnl[model_id] = session.model_contribution_to_pnl.get(model_id, 0.0) + trade_metrics.pnl_absolute
            
            # Update regime performance
            regime = trade_metrics.regime_type
            if regime not in session.regime_performance:
                session.regime_performance[regime] = {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0}
            
            session.regime_performance[regime]['trades'] += 1
            if trade_metrics.pnl_absolute:
                session.regime_performance[regime]['pnl'] += trade_metrics.pnl_absolute
            
            session.regime_frequency[regime] = session.regime_frequency.get(regime, 0) + 1
            
            # Calculate derived metrics
            if session.total_trades > 0:
                session.win_rate = session.winning_trades / session.total_trades
                session.avg_win = session.gross_profit / session.winning_trades if session.winning_trades > 0 else 0.0
                session.avg_loss = session.gross_loss / session.losing_trades if session.losing_trades > 0 else 0.0
                session.profit_factor = session.gross_profit / session.gross_loss if session.gross_loss > 0 else 0.0
            
            # Update execution quality
            if trade_metrics.execution_time_ms > 0:
                current_avg = session.avg_execution_time_ms
                n = session.total_trades
                session.avg_execution_time_ms = (current_avg * (n - 1) + trade_metrics.execution_time_ms) / n
            
            if trade_metrics.slippage is not None:
                current_avg = session.avg_slippage
                n = session.total_trades
                session.avg_slippage = (current_avg * (n - 1) + trade_metrics.slippage) / n
            
            tprint_info(f"📊 Updated session metrics: {session.total_trades} trades, {session.win_rate:.2%} win rate")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to update session metrics: {e}")
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def generate_performance_report(
        self,
        report_type: str = "session",
        export_format: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive performance report.
        
        Args:
            report_type: "session", "daily", "weekly", "monthly"
            export_format: "json", "csv", "html"
            
        Returns:
            Performance report dictionary
        """
        try:
            tprint_info(f"📊 Generating {report_type} performance report...")
            
            if report_type == "session":
                report = await self._generate_session_report()
            elif report_type == "daily":
                report = await self._generate_daily_report()
            else:
                tprint_warning(f"⚠️ Report type {report_type} not implemented yet")
                return None
            
            # Export report
            if report:
                await self._export_performance_report(report, report_type, export_format)
                tprint_success(f"✅ Generated {report_type} performance report")
            
            return report
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate performance report: {e}")
            return None
    
    async def _generate_session_report(self) -> Dict[str, Any]:
        """Generate session-level performance report."""
        try:
            if not self.current_session:
                return {}
            
            session = self.current_session
            
            # Calculate advanced metrics
            if len(self.completed_trades) > 1:
                pnl_series = [t.pnl_absolute for t in self.completed_trades if t.pnl_absolute is not None]
                if pnl_series:
                    returns = np.array(pnl_series)
                    session.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
                    
                    # Calculate max drawdown
                    cumulative_returns = np.cumsum(returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = (cumulative_returns - running_max)
                    session.max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            # Create comprehensive report
            report = {
                'session_info': {
                    'session_id': session.session_id,
                    'start_time': session.start_time.isoformat(),
                    'end_time': session.end_time.isoformat() if session.end_time else None,
                    'duration_hours': (datetime.now() - session.start_time).total_seconds() / 3600
                },
                'trade_statistics': {
                    'total_trades': session.total_trades,
                    'winning_trades': session.winning_trades,
                    'losing_trades': session.losing_trades,
                    'break_even_trades': session.break_even_trades,
                    'win_rate': session.win_rate,
                    'avg_win': session.avg_win,
                    'avg_loss': session.avg_loss,
                    'profit_factor': session.profit_factor
                },
                'performance_metrics': {
                    'total_pnl': session.total_pnl,
                    'gross_profit': session.gross_profit,
                    'gross_loss': session.gross_loss,
                    'max_drawdown': session.max_drawdown,
                    'sharpe_ratio': session.sharpe_ratio,
                    'sortino_ratio': session.sortino_ratio,
                    'calmar_ratio': session.calmar_ratio
                },
                'execution_quality': {
                    'avg_execution_time_ms': session.avg_execution_time_ms,
                    'avg_slippage': session.avg_slippage,
                    'execution_success_rate': session.execution_success_rate
                },
                'model_performance': {
                    'model_usage_frequency': session.model_usage_frequency,
                    'model_contribution_to_pnl': session.model_contribution_to_pnl,
                    'model_accuracy': session.model_accuracy
                },
                'regime_analysis': {
                    'regime_performance': session.regime_performance,
                    'regime_frequency': session.regime_frequency
                },
                'detailed_trades': [t.to_dict() for t in session.trades]
            }
            
            return report
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate session report: {e}")
            return {}
    
    async def _generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        try:
            today = datetime.now().date()
            daily_trades = [
                t for t in self.completed_trades 
                if t.timestamp.date() == today
            ]
            
            if not daily_trades:
                return {'message': 'No trades found for today'}
            
            # Calculate daily metrics
            total_pnl = sum(t.pnl_absolute for t in daily_trades if t.pnl_absolute is not None)
            winning_trades = len([t for t in daily_trades if t.pnl_absolute and t.pnl_absolute > 0])
            losing_trades = len([t for t in daily_trades if t.pnl_absolute and t.pnl_absolute < 0])
            
            report = {
                'date': today.isoformat(),
                'trade_count': len(daily_trades),
                'total_pnl': total_pnl,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / len(daily_trades) if daily_trades else 0.0,
                'trades': [t.to_dict() for t in daily_trades]
            }
            
            return report
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate daily report: {e}")
            return {}
    
    async def _export_performance_report(
        self,
        report: Dict[str, Any],
        report_type: str,
        export_format: str
    ):
        """Export performance report to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if export_format == "json":
                filename = f"{report_type}_report_{timestamp}.json"
                filepath = self.export_directory / filename
                
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                tprint_success(f"✅ Exported {report_type} report to {filepath}")
            
            elif export_format == "csv":
                # Export key metrics to CSV
                filename = f"{report_type}_metrics_{timestamp}.csv"
                filepath = self.export_directory / filename
                
                # Flatten report for CSV export
                flattened_data = self._flatten_report_for_csv(report)
                df = pd.DataFrame([flattened_data])
                df.to_csv(filepath, index=False)
                
                tprint_success(f"✅ Exported {report_type} metrics to {filepath}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to export performance report: {e}")
    
    def _flatten_report_for_csv(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested report dictionary for CSV export."""
        flattened = {}
        
        def flatten_dict(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, (list, tuple)):
                    flattened[new_key] = str(value)
                else:
                    flattened[new_key] = value
        
        flatten_dict(report)
        return flattened
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitor statistics."""
        return {
            'is_initialized': self.is_initialized,
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'current_session': self.current_session.session_id if self.current_session else None,
            'session_metrics': asdict(self.current_session) if self.current_session else None,
            'explanations_enabled': self.enable_explanations,
            'real_time_export_enabled': self.enable_real_time_export,
            'export_directory': str(self.export_directory)
        }
    
    async def stop(self):
        """Stop the comprehensive trade monitor."""
        try:
            tprint_info("🛑 Stopping Comprehensive Trade Monitor...")
            
            # End current session
            if self.current_session:
                self.current_session.end_time = datetime.now()
                
                # Generate final session report
                final_report = await self.generate_performance_report("session", "json")
                if final_report:
                    tprint_success("✅ Generated final session report")
            
            # Stop enhanced monitoring
            if self.enhanced_monitoring:
                await self.enhanced_monitoring.stop()
            
            # Clear active trades (move to completed)
            for trade_id, trade_metrics in self.active_trades.items():
                self.completed_trades.append(trade_metrics)
            self.active_trades.clear()
            
            tprint_success("✅ Comprehensive Trade Monitor stopped successfully")
            
        except Exception as e:
            tprint_error(f"❌ Error stopping Comprehensive Trade Monitor: {e}")

# Global instance
comprehensive_trade_monitor = ComprehensiveTradeMonitor()

# Convenience functions
async def initialize_comprehensive_monitoring(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize comprehensive trade monitoring."""
    global comprehensive_trade_monitor
    comprehensive_trade_monitor = ComprehensiveTradeMonitor(config)
    return await comprehensive_trade_monitor.initialize()

async def record_detailed_trade(
    trade_data: Dict[str, Any],
    models_used: Optional[Dict[str, Any]] = None,
    market_data: Optional[pd.DataFrame] = None
) -> str:
    """Record a detailed trade with comprehensive metrics."""
    return await comprehensive_trade_monitor.record_trade_decision(trade_data, models_used, market_data)

async def update_trade_outcome(trade_id: str, outcome_data: Dict[str, Any]) -> bool:
    """Update trade with outcome data."""
    return await comprehensive_trade_monitor.update_trade_outcome(trade_id, outcome_data)

async def generate_comprehensive_report(
    report_type: str = "session",
    export_format: str = "json"
) -> Optional[Dict[str, Any]]:
    """Generate comprehensive performance report."""
    return await comprehensive_trade_monitor.generate_performance_report(report_type, export_format)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
