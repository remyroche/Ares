#!/usr/bin/env python3
"""
Trade Decision Context Capture System

Captures comprehensive trade decision context including exchange, token, time, price,
and all relevant market conditions for enhanced monitoring and analysis.
"""

import time
import uuid
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from .enhanced_ml_monitoring import TradeContext, TradingIndicator, MLModelDecision, EnsembleDecision, TradingMode, ModelType
import logging

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

except ImportError:
    
    cp = None

@dataclass
class MarketConditions:
    """Comprehensive market conditions at decision time."""
    # Price information
    current_price: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    
    # Volume information
    current_volume: float
    volume_change_1h: float
    volume_avg_24h: float
    
    # Volatility metrics
    volatility_1h: float
    volatility_24h: float
    atr_14: float  # Average True Range
    
    # Technical indicators
    rsi_14: float
    macd_signal: float
    macd_histogram: float
    bollinger_position: float  # Position within Bollinger Bands
    adx_14: float  # Average Directional Index
    
    # Market microstructure
    bid_ask_spread: float
    order_book_imbalance: float
    market_depth: float
    
    # Sentiment indicators
    fear_greed_index: Optional[float] = None
    social_sentiment: Optional[float] = None
    news_sentiment: Optional[float] = None
    
    # Additional context
    timestamp: datetime = None
    exchange: str = "unknown"
    symbol: str = "unknown"

@dataclass
class HMMRegimeContext:
    """HMM regime context for trade decisions."""
    regime_id: str
    regime_name: str
    regime_probability: float
    regime_transition_probability: float
    regime_duration: int  # Number of periods in current regime
    regime_stability_score: float
    next_regime_probabilities: Dict[str, float]
    
    # Regime-specific metrics
    regime_volatility: float
    regime_trend_strength: float
    regime_momentum: float
    
    # Historical regime performance
    regime_win_rate: float
    regime_avg_return: float
    regime_sharpe_ratio: float

@dataclass
class TradingSignalContext:
    """Context for trading signals and indicators."""
    # Signal strength
    signal_strength: float  # -1 to 1
    signal_confidence: float  # 0 to 1
    signal_quality: float  # 0 to 1
    
    # Signal components
    trend_signal: float
    momentum_signal: float
    mean_reversion_signal: float
    volatility_signal: float
    volume_signal: float
    
    # Signal timing
    signal_freshness: float  # How recent the signal is
    signal_persistence: float  # How long the signal has been active
    
    # Risk signals
    risk_level: float  # 0 to 1
    drawdown_risk: float
    volatility_risk: float
    liquidity_risk: float

@dataclass
class ModelDecisionContext:
    """Context for individual model decisions."""
    model_id: str
    model_type: ModelType
    model_version: str
    
    # Prediction details
    prediction: float
    confidence: float
    uncertainty: float
    
    # Feature importance
    top_features: List[Tuple[str, float]]  # (feature_name, importance)
    feature_contributions: Dict[str, float]
    
    # Model performance context
    recent_accuracy: float
    recent_sharpe_ratio: float
    model_stability_score: float
    
    # Processing context
    processing_time_ms: float
    data_quality_score: float
    model_health_score: float

@dataclass
class EnsembleDecisionContext:
    """Context for ensemble decision making."""
    ensemble_id: str
    ensemble_type: str  # "weighted_average", "voting", "stacking", etc.
    
    # Ensemble composition
    model_weights: Dict[str, float]
    model_count: int
    active_model_count: int
    
    # Decision metrics
    consensus_score: float
    disagreement_level: float
    diversity_score: float
    
    # Performance context
    ensemble_accuracy: float
    ensemble_stability: float
    ensemble_confidence: float
    
    # Weight dynamics
    weight_stability: float
    last_rebalance: Optional[datetime] = None
    rebalance_frequency: Optional[str] = None

@dataclass
class ComprehensiveTradeContext:
    """Comprehensive trade decision context with all relevant information."""
    # Basic context
    decision_id: str
    timestamp: datetime
    trading_mode: TradingMode
    
    # Market context
    exchange: str
    symbol: str
    base_asset: str
    quote_asset: str
    
    # Price and volume context
    current_price: float
    current_volume: float
    price_history: List[float]  # Recent price history
    volume_history: List[float]  # Recent volume history
    
    # Market conditions
    market_conditions: MarketConditions
    
    # HMM regime context
    hmm_regime_context: Optional[HMMRegimeContext] = None
    
    # Trading signal context
    trading_signal_context: Optional[TradingSignalContext] = None
    
    # Model decision contexts
    model_decision_contexts: List[ModelDecisionContext] = None
    
    # Ensemble decision context
    ensemble_decision_context: Optional[EnsembleDecisionContext] = None
    
    # Additional metadata
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    strategy_id: Optional[str] = None
    risk_parameters: Optional[Dict[str, float]] = None

class TradeDecisionContextCapture:
    """
    Captures comprehensive trade decision context for enhanced monitoring.
    
    This system provides:
    1. Context capture (exchange, token, time, price)
    2. Market conditions analysis
    3. HMM regime context
    4. Trading signal context
    5. Model decision context
    6. Ensemble decision context
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trade decision context capture."""
        self.config = config
        self.logger = system_logger.getChild("TradeDecisionContextCapture")
        
        # Configuration
        self.capture_config = config.get("trade_decision_capture", {})
        self.enable_market_conditions = self.capture_config.get("enable_market_conditions", True)
        self.enable_hmm_context = self.capture_config.get("enable_hmm_context", True)
        self.enable_signal_context = self.capture_config.get("enable_signal_context", True)
        self.enable_model_context = self.capture_config.get("enable_model_context", True)
        self.enable_ensemble_context = self.capture_config.get("enable_ensemble_context", True)
        
        # Data sources
        self.market_data_source = self.capture_config.get("market_data_source", None)
        self.hmm_model_source = self.capture_config.get("hmm_model_source", None)
        self.signal_generator_source = self.capture_config.get("signal_generator_source", None)
        
        # Storage
        self.captured_contexts: List[ComprehensiveTradeContext] = []
        self.context_cache: Dict[str, Any] = {}
        
        self.logger.info("Trade Decision Context Capture initialized")
    
    @handles_errors(default_return=None, context="trade_decision_capture.capture_trade_context")
    async def capture_trade_context(
        self,
        exchange: str,
        symbol: str,
        trading_mode: TradingMode,
        current_price: float,
        current_volume: float,
        price_history: Optional[List[float]] = None,
        volume_history: Optional[List[float]] = None,
        market_data: Optional[pd.DataFrame] = None,
        hmm_model: Optional[Any] = None,
        signal_generator: Optional[Any] = None,
        models: Optional[Dict[str, Any]] = None,
        ensemble: Optional[Any] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ComprehensiveTradeContext]:
        """Capture comprehensive trade decision context."""
        try:
            start_time = time.time()
            
            # Generate decision ID
            decision_id = f"{trading_mode.value}_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            # Parse symbol
            base_asset, quote_asset = self._parse_symbol(symbol)
            
            # Capture market conditions
            market_conditions = None
            if self.enable_market_conditions:
                market_conditions = await self._capture_market_conditions(
                    exchange, symbol, current_price, current_volume, market_data
                )
            
            # Capture HMM regime context
            hmm_regime_context = None
            if self.enable_hmm_context and hmm_model:
                hmm_regime_context = await self._capture_hmm_regime_context(
                    hmm_model, market_data, current_price
                )
            
            # Capture trading signal context
            trading_signal_context = None
            if self.enable_signal_context and signal_generator:
                trading_signal_context = await self._capture_trading_signal_context(
                    signal_generator, market_data, current_price
                )
            
            # Capture model decision contexts
            model_decision_contexts = []
            if self.enable_model_context and models:
                model_decision_contexts = await self._capture_model_decision_contexts(
                    models, market_data, current_price
                )
            
            # Capture ensemble decision context
            ensemble_decision_context = None
            if self.enable_ensemble_context and ensemble:
                ensemble_decision_context = await self._capture_ensemble_decision_context(
                    ensemble, model_decision_contexts
                )
            
            # Create comprehensive context
            comprehensive_context = ComprehensiveTradeContext(
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc),
                trading_mode=trading_mode,
                exchange=exchange,
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                current_price=current_price,
                current_volume=current_volume,
                price_history=price_history or [],
                volume_history=volume_history or [],
                market_conditions=market_conditions,
                hmm_regime_context=hmm_regime_context,
                trading_signal_context=trading_signal_context,
                model_decision_contexts=model_decision_contexts,
                ensemble_decision_context=ensemble_decision_context,
                session_id=additional_context.get('session_id') if additional_context else None,
                user_id=additional_context.get('user_id') if additional_context else None,
                strategy_id=additional_context.get('strategy_id') if additional_context else None,
                risk_parameters=additional_context.get('risk_parameters') if additional_context else None
            )
            
            # Store context
            self.captured_contexts.append(comprehensive_context)
            
            # Maintain memory limit
            if len(self.captured_contexts) > 10000:
                self.captured_contexts = self.captured_contexts[-10000:]
            
            capture_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"Captured trade context {decision_id} in {capture_time:.2f}ms: "
                f"{exchange}:{symbol} at {current_price}"
            )
            
            return comprehensive_context
            
        except Exception as e:
            self.logger.error(f"Error capturing trade context: {e}")
            return None
    
    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse trading symbol into base and quote assets."""
        try:
            # Common patterns: BTCUSDT, BTC/USDT, BTC-USDT
            if '/' in symbol:
                base, quote = symbol.split('/', 1)
            elif '-' in symbol:
                base, quote = symbol.split('-', 1)
            else:
                # Try to find common quote currencies
                quote_currencies = ['USDT', 'USDC', 'BTC', 'ETH', 'USD', 'EUR']
                for quote in quote_currencies:
                    if symbol.endswith(quote):
                        base = symbol[:-len(quote)]
                        return base, quote
                
                # Default parsing
                base = symbol[:-3] if len(symbol) > 3 else symbol
                quote = symbol[-3:] if len(symbol) > 3 else 'USD'
            
            return base, quote
            
        except Exception as e:
            self.logger.error(f"Error parsing symbol {symbol}: {e}")
            return symbol, 'USD'
    
    async def _capture_market_conditions(
        self,
        exchange: str,
        symbol: str,
        current_price: float,
        current_volume: float,
        market_data: Optional[pd.DataFrame]
    ) -> Optional[MarketConditions]:
        """Capture comprehensive market conditions."""
        try:
            if market_data is None or market_data.empty:
                # Create basic market conditions with available data
                return MarketConditions(
                    current_price=current_price,
                    price_change_1h=0.0,
                    price_change_24h=0.0,
                    price_change_7d=0.0,
                    current_volume=current_volume,
                    volume_change_1h=0.0,
                    volume_avg_24h=current_volume,
                    volatility_1h=0.0,
                    volatility_24h=0.0,
                    atr_14=0.0,
                    rsi_14=50.0,
                    macd_signal=0.0,
                    macd_histogram=0.0,
                    bollinger_position=0.5,
                    adx_14=25.0,
                    bid_ask_spread=0.0,
                    order_book_imbalance=0.0,
                    market_depth=0.0,
                    timestamp=datetime.now(timezone.utc),
                    exchange=exchange,
                    symbol=symbol
                )
            
            # Calculate price changes
            price_change_1h = self._calculate_price_change(market_data, '1h')
            price_change_24h = self._calculate_price_change(market_data, '24h')
            price_change_7d = self._calculate_price_change(market_data, '7d')
            
            # Calculate volume changes
            volume_change_1h = self._calculate_volume_change(market_data, '1h')
            volume_avg_24h = self._calculate_volume_average(market_data, '24h')
            
            # Calculate volatility
            volatility_1h = self._calculate_volatility(market_data, '1h')
            volatility_24h = self._calculate_volatility(market_data, '24h')
            atr_14 = self._calculate_atr(market_data, 14)
            
            # Calculate technical indicators
            rsi_14 = self._calculate_rsi(market_data, 14)
            macd_signal, macd_histogram = self._calculate_macd(market_data)
            bollinger_position = self._calculate_bollinger_position(market_data)
            adx_14 = self._calculate_adx(market_data, 14)
            
            # Market microstructure (would need order book data)
            bid_ask_spread = 0.0  # Would need real order book data
            order_book_imbalance = 0.0  # Would need real order book data
            market_depth = 0.0  # Would need real order book data
            
            return MarketConditions(
                current_price=current_price,
                price_change_1h=price_change_1h,
                price_change_24h=price_change_24h,
                price_change_7d=price_change_7d,
                current_volume=current_volume,
                volume_change_1h=volume_change_1h,
                volume_avg_24h=volume_avg_24h,
                volatility_1h=volatility_1h,
                volatility_24h=volatility_24h,
                atr_14=atr_14,
                rsi_14=rsi_14,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bollinger_position=bollinger_position,
                adx_14=adx_14,
                bid_ask_spread=bid_ask_spread,
                order_book_imbalance=order_book_imbalance,
                market_depth=market_depth,
                timestamp=datetime.now(timezone.utc),
                exchange=exchange,
                symbol=symbol
            )
            
        except Exception as e:
            self.logger.error(f"Error capturing market conditions: {e}")
            return None
    
    async def _capture_hmm_regime_context(
        self,
        hmm_model: Any,
        market_data: Optional[pd.DataFrame],
        current_price: float
    ) -> Optional[HMMRegimeContext]:
        """Capture HMM regime context."""
        try:
            if not hasattr(hmm_model, 'predict') or market_data is None:
                return None
            
            # Get current regime
            current_regime = hmm_model.predict([current_price])[0] if hasattr(hmm_model, 'predict') else 0
            
            # Get regime probabilities
            regime_probs = hmm_model.predict_proba([current_price])[0] if hasattr(hmm_model, 'predict_proba') else [1.0]
            
            # Calculate regime metrics
            regime_probability = float(regime_probs[current_regime]) if current_regime < len(regime_probs) else 0.0
            
            # Calculate transition probabilities (would need more sophisticated HMM model)
            regime_transition_probability = 0.1  # Placeholder
            
            # Calculate regime duration (would need historical regime data)
            regime_duration = 1  # Placeholder
            
            # Calculate regime stability
            regime_stability_score = regime_probability
            
            # Next regime probabilities
            next_regime_probabilities = {f"regime_{i}": float(prob) for i, prob in enumerate(regime_probs)}
            
            # Regime-specific metrics (would need historical data)
            regime_volatility = 0.0
            regime_trend_strength = 0.0
            regime_momentum = 0.0
            regime_win_rate = 0.5
            regime_avg_return = 0.0
            regime_sharpe_ratio = 0.0
            
            return HMMRegimeContext(
                regime_id=f"regime_{current_regime}",
                regime_name=f"Regime {current_regime}",
                regime_probability=regime_probability,
                regime_transition_probability=regime_transition_probability,
                regime_duration=regime_duration,
                regime_stability_score=regime_stability_score,
                next_regime_probabilities=next_regime_probabilities,
                regime_volatility=regime_volatility,
                regime_trend_strength=regime_trend_strength,
                regime_momentum=regime_momentum,
                regime_win_rate=regime_win_rate,
                regime_avg_return=regime_avg_return,
                regime_sharpe_ratio=regime_sharpe_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Error capturing HMM regime context: {e}")
            return None
    
    async def _capture_trading_signal_context(
        self,
        signal_generator: Any,
        market_data: Optional[pd.DataFrame],
        current_price: float
    ) -> Optional[TradingSignalContext]:
        """Capture trading signal context."""
        try:
            if not hasattr(signal_generator, 'generate_signals') or market_data is None:
                return None
            
            # Generate signals
            signals = signal_generator.generate_signals(market_data, current_price)
            
            # Extract signal components
            signal_strength = signals.get('signal_strength', 0.0)
            signal_confidence = signals.get('signal_confidence', 0.5)
            signal_quality = signals.get('signal_quality', 0.5)
            
            # Signal components
            trend_signal = signals.get('trend_signal', 0.0)
            momentum_signal = signals.get('momentum_signal', 0.0)
            mean_reversion_signal = signals.get('mean_reversion_signal', 0.0)
            volatility_signal = signals.get('volatility_signal', 0.0)
            volume_signal = signals.get('volume_signal', 0.0)
            
            # Signal timing
            signal_freshness = signals.get('signal_freshness', 1.0)
            signal_persistence = signals.get('signal_persistence', 0.0)
            
            # Risk signals
            risk_level = signals.get('risk_level', 0.5)
            drawdown_risk = signals.get('drawdown_risk', 0.0)
            volatility_risk = signals.get('volatility_risk', 0.0)
            liquidity_risk = signals.get('liquidity_risk', 0.0)
            
            return TradingSignalContext(
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
                signal_quality=signal_quality,
                trend_signal=trend_signal,
                momentum_signal=momentum_signal,
                mean_reversion_signal=mean_reversion_signal,
                volatility_signal=volatility_signal,
                volume_signal=volume_signal,
                signal_freshness=signal_freshness,
                signal_persistence=signal_persistence,
                risk_level=risk_level,
                drawdown_risk=drawdown_risk,
                volatility_risk=volatility_risk,
                liquidity_risk=liquidity_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error capturing trading signal context: {e}")
            return None
    
    async def _capture_model_decision_contexts(
        self,
        models: Dict[str, Any],
        market_data: Optional[pd.DataFrame],
        current_price: float
    ) -> List[ModelDecisionContext]:
        """Capture model decision contexts."""
        try:
            model_contexts = []
            
            for model_id, model_info in models.items():
                model = model_info.get('model')
                model_type = model_info.get('type', 'unknown')
                model_version = model_info.get('version', '1.0')
                
                if not model or not hasattr(model, 'predict'):
                    continue
                
                # Get prediction
                prediction = model.predict([current_price])[0] if hasattr(model, 'predict') else 0.0
                
                # Get confidence (if available)
                confidence = model_info.get('confidence', 0.5)
                uncertainty = model_info.get('uncertainty', 0.5)
                
                # Get feature importance (if available)
                top_features = model_info.get('top_features', [])
                feature_contributions = model_info.get('feature_contributions', {})
                
                # Model performance context
                recent_accuracy = model_info.get('recent_accuracy', 0.5)
                recent_sharpe_ratio = model_info.get('recent_sharpe_ratio', 0.0)
                model_stability_score = model_info.get('stability_score', 0.5)
                
                # Processing context
                processing_time_ms = model_info.get('processing_time_ms', 0.0)
                data_quality_score = model_info.get('data_quality_score', 1.0)
                model_health_score = model_info.get('health_score', 1.0)
                
                model_context = ModelDecisionContext(
                    model_id=model_id,
                    model_type=ModelType(model_type) if model_type in [e.value for e in ModelType] else ModelType.HMM,
                    model_version=model_version,
                    prediction=prediction,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    top_features=top_features,
                    feature_contributions=feature_contributions,
                    recent_accuracy=recent_accuracy,
                    recent_sharpe_ratio=recent_sharpe_ratio,
                    model_stability_score=model_stability_score,
                    processing_time_ms=processing_time_ms,
                    data_quality_score=data_quality_score,
                    model_health_score=model_health_score
                )
                
                model_contexts.append(model_context)
            
            return model_contexts
            
        except Exception as e:
            self.logger.error(f"Error capturing model decision contexts: {e}")
            return []
    
    async def _capture_ensemble_decision_context(
        self,
        ensemble: Any,
        model_decision_contexts: List[ModelDecisionContext]
    ) -> Optional[EnsembleDecisionContext]:
        """Capture ensemble decision context."""
        try:
            if not ensemble or not model_decision_contexts:
                return None
            
            # Get ensemble information
            ensemble_id = getattr(ensemble, 'ensemble_id', 'default_ensemble')
            ensemble_type = getattr(ensemble, 'ensemble_type', 'weighted_average')
            
            # Calculate model weights
            model_weights = {}
            total_weight = 0.0
            
            for model_context in model_decision_contexts:
                weight = model_context.confidence * model_context.model_stability_score
                model_weights[model_context.model_id] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                model_weights = {k: v / total_weight for k, v in model_weights.items()}
            
            # Calculate ensemble metrics
            model_count = len(model_decision_contexts)
            active_model_count = sum(1 for w in model_weights.values() if w > 0.01)
            
            # Calculate consensus and disagreement
            predictions = [mc.prediction for mc in model_decision_contexts]
            consensus_score = 1.0 - np.std(predictions) if len(predictions) > 1 else 1.0
            disagreement_level = np.std(predictions) if len(predictions) > 1 else 0.0
            
            # Calculate diversity score
            diversity_score = len(set(predictions)) / len(predictions) if predictions else 0.0
            
            # Ensemble performance context
            ensemble_accuracy = np.mean([mc.recent_accuracy for mc in model_decision_contexts])
            ensemble_stability = np.mean([mc.model_stability_score for mc in model_decision_contexts])
            ensemble_confidence = np.mean([mc.confidence for mc in model_decision_contexts])
            
            # Weight dynamics
            weight_stability = 1.0  # Would need historical data
            last_rebalance = None  # Would need historical data
            rebalance_frequency = None  # Would need historical data
            
            return EnsembleDecisionContext(
                ensemble_id=ensemble_id,
                ensemble_type=ensemble_type,
                model_weights=model_weights,
                model_count=model_count,
                active_model_count=active_model_count,
                consensus_score=consensus_score,
                disagreement_level=disagreement_level,
                diversity_score=diversity_score,
                ensemble_accuracy=ensemble_accuracy,
                ensemble_stability=ensemble_stability,
                ensemble_confidence=ensemble_confidence,
                weight_stability=weight_stability,
                last_rebalance=last_rebalance,
                rebalance_frequency=rebalance_frequency
            )
            
        except Exception as e:
            self.logger.error(f"Error capturing ensemble decision context: {e}")
            return None
    
    # Helper methods for technical indicators
    def _calculate_price_change(self, data: pd.DataFrame, period: str) -> float:
        """Calculate price change for given period."""
        try:
            if 'close' not in data.columns or len(data) < 2:
                return 0.0
            
            current_price = data['close'].iloc[-1]
            
            if period == '1h' and len(data) >= 4:  # Assuming 15-min data
                past_price = data['close'].iloc[-4]
            elif period == '24h' and len(data) >= 96:  # Assuming 15-min data
                past_price = data['close'].iloc[-96]
            elif period == '7d' and len(data) >= 672:  # Assuming 15-min data
                past_price = data['close'].iloc[-672]
            else:
                past_price = data['close'].iloc[0]
            
            return (current_price - past_price) / past_price if past_price > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volume_change(self, data: pd.DataFrame, period: str) -> float:
        """Calculate volume change for given period."""
        try:
            if 'volume' not in data.columns or len(data) < 2:
                return 0.0
            
            current_volume = data['volume'].iloc[-1]
            
            if period == '1h' and len(data) >= 4:
                past_volume = data['volume'].iloc[-4]
            else:
                past_volume = data['volume'].iloc[0]
            
            return (current_volume - past_volume) / past_volume if past_volume > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volume_average(self, data: pd.DataFrame, period: str) -> float:
        """Calculate average volume for given period."""
        try:
            if 'volume' not in data.columns:
                return 0.0
            
            if period == '24h' and len(data) >= 96:
                return data['volume'].iloc[-96:].mean()
            else:
                return data['volume'].mean()
                
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, data: pd.DataFrame, period: str) -> float:
        """Calculate volatility for given period."""
        try:
            if 'close' not in data.columns or len(data) < 2:
                return 0.0
            
            if period == '1h' and len(data) >= 4:
                returns = data['close'].iloc[-4:].pct_change().dropna()
            elif period == '24h' and len(data) >= 96:
                returns = data['close'].iloc[-96:].pct_change().dropna()
            else:
                returns = data['close'].pct_change().dropna()
            
            return returns.std() if len(returns) > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range."""
        try:
            if len(data.columns) < 3 or len(data) < period:
                return 0.0
            
            high = data['high'] if 'high' in data.columns else data['close']
            low = data['low'] if 'low' in data.columns else data['close']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            return self._vectorbt_rolling_operation(tr, "mean", period).iloc[-1] if len(tr) >= period else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Relative Strength Index."""
        try:
            if 'close' not in data.columns or len(data) < period + 1:
                return 50.0
            
            close = data['close']
            delta = close.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_macd(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate MACD signal and histogram."""
        try:
            if 'close' not in data.columns or len(data) < 26:
                return 0.0, 0.0
            
            close = data['close']
            
            # Calculate MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            return float(signal_line.iloc[-1]), float(histogram.iloc[-1])
            
        except Exception:
            return 0.0, 0.0
    
    def _calculate_bollinger_position(self, data: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands."""
        try:
            if 'close' not in data.columns or len(data) < 20:
                return 0.5
            
            close = data['close']
            sma_20 = self._vectorbt_rolling_operation(close, "mean", 20)
            std_20 = self._vectorbt_rolling_operation(close, "std", 20)
            
            upper_band = sma_20 + (2 * std_20)
            lower_band = sma_20 - (2 * std_20)
            
            current_price = close.iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]
            
            if upper == lower:
                return 0.5
            
            position = (current_price - lower) / (upper - lower)
            return max(0.0, min(1.0, position))
            
        except Exception:
            return 0.5
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Average Directional Index."""
        try:
            if len(data.columns) < 3 or len(data) < period:
                return 25.0
            
            high = data['high'] if 'high' in data.columns else data['close']
            low = data['low'] if 'low' in data.columns else data['close']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Calculate smoothed values
            tr_smooth = self._vectorbt_rolling_operation(tr, "mean", period)
            dm_plus_smooth = self._vectorbt_rolling_operation(dm_plus, "mean", period)
            dm_minus_smooth = self._vectorbt_rolling_operation(dm_minus, "mean", period)
            
            # Calculate DI
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = self._vectorbt_rolling_operation(dx, "mean", period)
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
            
        except Exception:
            return 25.0
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get statistics about context capture."""
        return {
            'total_contexts_captured': len(self.captured_contexts),
            'enable_market_conditions': self.enable_market_conditions,
            'enable_hmm_context': self.enable_hmm_context,
            'enable_signal_context': self.enable_signal_context,
            'enable_model_context': self.enable_model_context,
            'enable_ensemble_context': self.enable_ensemble_context,
            'context_cache_size': len(self.context_cache)
        }