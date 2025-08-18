"""
Example Usage of Trading Decorators

This file demonstrates how to apply the new trading decorators to your
existing trading and backtesting pipelines for enhanced error handling,
trade tracking, monitoring, and operational management.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from src.utils.trading_decorators import (
    # Error handling decorators
    trading_error_handler,
    market_data_error_handler,
    
    # Trade tracking decorators
    track_trade,
    track_model_performance,
    
    # Performance monitoring decorators
    monitor_performance,
    validate_trade_parameters,
    
    # Operational decorators
    rate_limit,
    circuit_breaker,
    retry_with_backoff,
    
    # Composite decorators
    comprehensive_trade_decorator,
    comprehensive_model_decorator,
    
    # Utility functions
    get_trade_tracker,
    TradeSide,
    ExecutionMode
)


# ============================================================================
# EXAMPLE 1: Enhanced Backtesting Pipeline
# ============================================================================

class EnhancedBacktesterWithDecorators:
    """
    Example of how to enhance your existing backtester with decorators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trade_tracker = get_trade_tracker()
    
    @comprehensive_trade_decorator(
        enable_error_handling=True,
        enable_tracking=True,
        enable_performance_monitoring=True,
        enable_validation=True,
        enable_rate_limiting=False,  # Not needed for backtesting
        enable_circuit_breaker=True,
        retry_attempts=3,
        alert_threshold_ms=5000.0  # 5 seconds for backtesting
    )
    async def execute_backtest_trade(
        self,
        symbol: str,
        signal: int,
        price: float,
        timestamp: datetime,
        trade_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a trade during backtesting with comprehensive tracking.
        
        This method will automatically:
        - Validate trade parameters
        - Track all trade data (model weights, confidences, regime analysis)
        - Monitor performance
        - Handle errors with retries and circuit breaker
        - Log everything to the trade tracker
        """
        if trade_metadata is None:
            trade_metadata = {}
        
        # Extract model data from metadata
        model_weights = trade_metadata.get('model_weights', {})
        model_confidences = trade_metadata.get('model_confidences', {})
        regime_analysis = trade_metadata.get('regime_analysis', {})
        hmm_regime = trade_metadata.get('hmm_regime', '')
        support_resistance = trade_metadata.get('support_resistance_levels', {})
        market_conditions = trade_metadata.get('market_conditions', {})
        risk_metrics = trade_metadata.get('risk_metrics', {})
        
        # Calculate position size
        position_size = self.config.get('portfolio_value', 10000.0) * 0.1
        quantity = position_size / price
        
        # Execute trade based on signal
        if signal == 1:  # Buy signal
            side = TradeSide.BUY
            trade_result = await self._execute_buy_trade(
                symbol=symbol,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                model_weights=model_weights,
                model_confidences=model_confidences,
                regime_analysis=regime_analysis,
                hmm_regime=hmm_regime,
                support_resistance_levels=support_resistance,
                market_conditions=market_conditions,
                risk_metrics=risk_metrics
            )
        elif signal == -1:  # Sell signal
            side = TradeSide.SELL
            trade_result = await self._execute_sell_trade(
                symbol=symbol,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                model_weights=model_weights,
                model_confidences=model_confidences,
                regime_analysis=regime_analysis,
                hmm_regime=hmm_regime,
                support_resistance_levels=support_resistance,
                market_conditions=market_conditions,
                risk_metrics=risk_metrics
            )
        else:
            return None
        
        return trade_result
    
    @track_trade(
        capture_model_data=True,
        capture_regime_data=True,
        capture_market_conditions=True,
        capture_risk_metrics=True
    )
    async def _execute_buy_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        model_weights: Dict[str, float],
        model_confidences: Dict[str, float],
        regime_analysis: Dict[str, Any],
        hmm_regime: str,
        support_resistance_levels: Dict[str, float],
        market_conditions: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute a buy trade with comprehensive tracking."""
        
        # Simulate trade execution
        trade_id = f"BUY_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate costs
        total_cost = quantity * price
        commission = total_cost * 0.001
        total_with_fees = total_cost + commission
        
        trade_result = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': 'buy',
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'total_cost': total_with_fees,
            'commission': commission,
            'execution_mode': ExecutionMode.BACKTEST.value,
            'model_weights': model_weights,
            'model_confidences': model_confidences,
            'regime_analysis': regime_analysis,
            'hmm_regime': hmm_regime,
            'support_resistance_levels': support_resistance_levels,
            'market_conditions': market_conditions,
            'risk_metrics': risk_metrics
        }
        
        return trade_result
    
    @track_trade(
        capture_model_data=True,
        capture_regime_data=True,
        capture_market_conditions=True,
        capture_risk_metrics=True
    )
    async def _execute_sell_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        model_weights: Dict[str, float],
        model_confidences: Dict[str, float],
        regime_analysis: Dict[str, Any],
        hmm_regime: str,
        support_resistance_levels: Dict[str, float],
        market_conditions: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute a sell trade with comprehensive tracking."""
        
        trade_id = f"SELL_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate proceeds
        total_proceeds = quantity * price
        commission = total_proceeds * 0.001
        net_proceeds = total_proceeds - commission
        
        trade_result = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': 'sell',
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'total_proceeds': net_proceeds,
            'commission': commission,
            'execution_mode': ExecutionMode.BACKTEST.value,
            'model_weights': model_weights,
            'model_confidences': model_confidences,
            'regime_analysis': regime_analysis,
            'hmm_regime': hmm_regime,
            'support_resistance_levels': support_resistance_levels,
            'market_conditions': market_conditions,
            'risk_metrics': risk_metrics
        }
        
        return trade_result


# ============================================================================
# EXAMPLE 2: Enhanced Live Trading Pipeline
# ============================================================================

class LiveTradingPipelineWithDecorators:
    """
    Example of how to enhance your existing live trading pipeline with decorators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trade_tracker = get_trade_tracker()
    
    @comprehensive_trade_decorator(
        enable_error_handling=True,
        enable_tracking=True,
        enable_performance_monitoring=True,
        enable_validation=True,
        enable_rate_limiting=True,
        enable_circuit_breaker=True,
        max_calls=50,  # Rate limit for live trading
        time_window=60.0,
        alert_threshold_ms=2000.0  # 2 seconds for live trading
    )
    async def execute_live_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = 'market',
        model_weights: Dict[str, float] = None,
        model_confidences: Dict[str, float] = None,
        regime_analysis: Dict[str, Any] = None,
        hmm_regime: str = '',
        support_resistance_levels: Dict[str, float] = None,
        market_conditions: Dict[str, Any] = None,
        risk_metrics: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Execute a live trade with comprehensive tracking and safety measures.
        
        This method will automatically:
        - Validate all trade parameters
        - Rate limit API calls
        - Track all trade data
        - Monitor performance
        - Handle errors with circuit breaker
        - Log everything for monitoring
        """
        
        # Prepare trade metadata
        trade_metadata = {
            'model_weights': model_weights or {},
            'model_confidences': model_confidences or {},
            'regime_analysis': regime_analysis or {},
            'hmm_regime': hmm_regime,
            'support_resistance_levels': support_resistance_levels or {},
            'market_conditions': market_conditions or {},
            'risk_metrics': risk_metrics or {},
            'order_type': order_type
        }
        
        # Execute the trade
        if side.lower() == 'buy':
            return await self._place_buy_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                **trade_metadata
            )
        elif side.lower() == 'sell':
            return await self._place_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                **trade_metadata
            )
        else:
            raise ValueError(f"Invalid trade side: {side}")
    
    @track_trade(
        capture_model_data=True,
        capture_regime_data=True,
        capture_market_conditions=True,
        capture_risk_metrics=True
    )
    async def _place_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        order_type: str,
        model_weights: Dict[str, float],
        model_confidences: Dict[str, float],
        regime_analysis: Dict[str, Any],
        hmm_regime: str,
        support_resistance_levels: Dict[str, float],
        market_conditions: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Place a buy order with comprehensive tracking."""
        
        # Simulate order placement
        order_id = f"BUY_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # In real implementation, this would call your exchange API
        # order_result = await self.exchange_client.place_order(...)
        
        order_result = {
            'order_id': order_id,
            'symbol': symbol,
            'side': 'buy',
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'status': 'filled',
            'timestamp': datetime.now(),
            'execution_mode': ExecutionMode.LIVE.value,
            'model_weights': model_weights,
            'model_confidences': model_confidences,
            'regime_analysis': regime_analysis,
            'hmm_regime': hmm_regime,
            'support_resistance_levels': support_resistance_levels,
            'market_conditions': market_conditions,
            'risk_metrics': risk_metrics
        }
        
        return order_result
    
    @track_trade(
        capture_model_data=True,
        capture_regime_data=True,
        capture_market_conditions=True,
        capture_risk_metrics=True
    )
    async def _place_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        order_type: str,
        model_weights: Dict[str, float],
        model_confidences: Dict[str, float],
        regime_analysis: Dict[str, Any],
        hmm_regime: str,
        support_resistance_levels: Dict[str, float],
        market_conditions: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Place a sell order with comprehensive tracking."""
        
        order_id = f"SELL_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        order_result = {
            'order_id': order_id,
            'symbol': symbol,
            'side': 'sell',
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'status': 'filled',
            'timestamp': datetime.now(),
            'execution_mode': ExecutionMode.LIVE.value,
            'model_weights': model_weights,
            'model_confidences': model_confidences,
            'regime_analysis': regime_analysis,
            'hmm_regime': hmm_regime,
            'support_resistance_levels': support_resistance_levels,
            'market_conditions': market_conditions,
            'risk_metrics': risk_metrics
        }
        
        return order_result


# ============================================================================
# EXAMPLE 3: Enhanced Model Operations
# ============================================================================

class ModelOperationsWithDecorators:
    """
    Example of how to enhance model operations with decorators.
    """
    
    @comprehensive_model_decorator(
        enable_error_handling=True,
        enable_tracking=True,
        enable_performance_monitoring=True,
        enable_retry=True,
        model_name="XGBoost_Ensemble",
        capture_predictions=True,
        capture_feature_importance=True,
        capture_confidence=True,
        retry_attempts=3,
        alert_threshold_ms=3000.0
    )
    async def predict_with_ensemble(
        self,
        features: pd.DataFrame,
        model_weights: Dict[str, float],
        regime_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make predictions with ensemble model and comprehensive tracking.
        
        This method will automatically:
        - Track model performance
        - Monitor execution time
        - Handle errors with retries
        - Log predictions and feature importance
        """
        
        # Simulate ensemble prediction
        predictions = {}
        confidences = {}
        feature_importance = {}
        
        for model_name, weight in model_weights.items():
            # Simulate individual model prediction
            pred = self._simulate_model_prediction(model_name, features)
            predictions[model_name] = pred['prediction']
            confidences[model_name] = pred['confidence']
            feature_importance[model_name] = pred['feature_importance']
        
        # Calculate ensemble prediction
        ensemble_prediction = self._calculate_ensemble_prediction(
            predictions, confidences, model_weights
        )
        
        result = {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'model_weights': model_weights,
            'feature_importance': feature_importance,
            'regime_analysis': regime_analysis or {},
            'timestamp': datetime.now()
        }
        
        return result
    
    @track_model_performance(
        model_name="XGBoost_Individual",
        capture_predictions=True,
        capture_feature_importance=True,
        capture_confidence=True
    )
    def _simulate_model_prediction(
        self,
        model_name: str,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Simulate individual model prediction with tracking."""
        
        # Simulate prediction logic
        import numpy as np
        
        prediction = np.random.choice(['buy', 'sell', 'hold'])
        confidence = np.random.uniform(0.5, 0.95)
        feature_importance = {
            'feature_1': np.random.uniform(0.1, 0.3),
            'feature_2': np.random.uniform(0.1, 0.3),
            'feature_3': np.random.uniform(0.1, 0.3)
        }
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'feature_importance': feature_importance,
            'model_name': model_name
        }
    
    def _calculate_ensemble_prediction(
        self,
        predictions: Dict[str, str],
        confidences: Dict[str, float],
        weights: Dict[str, float]
    ) -> str:
        """Calculate ensemble prediction from individual model predictions."""
        
        # Simple weighted voting
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 1.0)
            confidence = confidences.get(model_name, 0.5)
            score = weight * confidence
            
            if prediction == 'buy':
                buy_score += score
            elif prediction == 'sell':
                sell_score += score
            else:  # hold
                hold_score += score
        
        # Return prediction with highest score
        scores = {'buy': buy_score, 'sell': sell_score, 'hold': hold_score}
        return max(scores, key=scores.get)


# ============================================================================
# EXAMPLE 4: Market Data Operations
# ============================================================================

class MarketDataOperationsWithDecorators:
    """
    Example of how to enhance market data operations with decorators.
    """
    
    @market_data_error_handler(
        data_validation=True,
        fallback_to_cached=True,
        max_age_seconds=300
    )
    @monitor_performance(
        alert_threshold_ms=1000.0,
        log_slow_operations=True,
        capture_memory_usage=False
    )
    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True
    )
    async def fetch_market_data(
        self,
        symbol: str,
        interval: str = '1m',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch market data with comprehensive error handling and monitoring.
        
        This method will automatically:
        - Validate data quality
        - Fallback to cached data if needed
        - Monitor performance
        - Retry on failures with backoff
        - Log all operations
        """
        
        # Simulate market data fetch
        # In real implementation, this would call your exchange API
        # data = await self.exchange_client.get_klines(symbol, interval, limit)
        
        # Simulate data
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, limit),
            'high': np.random.uniform(100, 200, limit),
            'low': np.random.uniform(100, 200, limit),
            'close': np.random.uniform(100, 200, limit),
            'volume': np.random.uniform(1000, 10000, limit)
        }, index=dates)
        
        return data
    
    @circuit_breaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        monitor_interval=10.0
    )
    @rate_limit(
        max_calls=100,
        time_window=60.0
    )
    async def fetch_real_time_data(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Fetch real-time market data with circuit breaker and rate limiting.
        
        This method will automatically:
        - Rate limit API calls
        - Open circuit breaker on repeated failures
        - Recover automatically after timeout
        """
        
        # Simulate real-time data fetch
        data = {
            'symbol': symbol,
            'price': 150.0 + (datetime.now().microsecond % 100) / 100,
            'volume': 5000 + (datetime.now().microsecond % 1000),
            'timestamp': datetime.now(),
            'bid': 149.95,
            'ask': 150.05
        }
        
        return data


# ============================================================================
# EXAMPLE 5: Integration with Existing Pipelines
# ============================================================================

async def integrate_with_existing_backtester():
    """
    Example of how to integrate decorators with your existing backtester.
    """
    
    # Create enhanced backtester
    config = {
        'portfolio_value': 10000.0,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005
    }
    
    backtester = EnhancedBacktesterWithDecorators(config)
    
    # Simulate strategy signals
    signals = pd.DataFrame({
        'signal': [1, 0, -1, 1, 0],  # Buy, Hold, Sell, Buy, Hold
        'close': [150.0, 151.0, 152.0, 153.0, 154.0]
    })
    
    # Simulate trade metadata with model data
    trade_metadata = {
        'model_weights': {
            'xgboost': 0.4,
            'lstm': 0.3,
            'random_forest': 0.3
        },
        'model_confidences': {
            'xgboost': 0.85,
            'lstm': 0.78,
            'random_forest': 0.82
        },
        'regime_analysis': {
            'regime_type': 'trending',
            'regime_confidence': 0.75,
            'volatility': 0.15
        },
        'hmm_regime': 'bull_market',
        'support_resistance_levels': {
            'support': 148.0,
            'resistance': 155.0
        },
        'market_conditions': {
            'trend': 'upward',
            'volume': 'high',
            'volatility': 'medium'
        },
        'risk_metrics': {
            'var_95': 0.02,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.2
        }
    }
    
    # Execute trades with comprehensive tracking
    for i, (timestamp, row) in enumerate(signals.iterrows()):
        if row['signal'] != 0:
            result = await backtester.execute_backtest_trade(
                symbol='BTCUSDT',
                signal=row['signal'],
                price=row['close'],
                timestamp=timestamp,
                trade_metadata=trade_metadata
            )
            print(f"Trade {i+1}: {result}")
    
    # Get trade tracking data
    tracker = get_trade_tracker()
    print(f"Total trades tracked: {len(tracker.trades)}")
    print(f"Performance history: {len(tracker.performance_history)}")


async def integrate_with_existing_live_trader():
    """
    Example of how to integrate decorators with your existing live trader.
    """
    
    # Create enhanced live trader
    config = {
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret',
        'testnet': True
    }
    
    live_trader = LiveTradingPipelineWithDecorators(config)
    
    # Execute live trade with comprehensive tracking
    result = await live_trader.execute_live_trade(
        symbol='BTCUSDT',
        side='buy',
        quantity=0.001,
        price=150.0,
        order_type='market',
        model_weights={
            'xgboost': 0.4,
            'lstm': 0.3,
            'random_forest': 0.3
        },
        model_confidences={
            'xgboost': 0.85,
            'lstm': 0.78,
            'random_forest': 0.82
        },
        regime_analysis={
            'regime_type': 'trending',
            'regime_confidence': 0.75
        },
        hmm_regime='bull_market',
        support_resistance_levels={
            'support': 148.0,
            'resistance': 155.0
        },
        market_conditions={
            'trend': 'upward',
            'volume': 'high'
        },
        risk_metrics={
            'var_95': 0.02,
            'max_drawdown': 0.05
        }
    )
    
    print(f"Live trade result: {result}")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    """Run all examples."""
    print("=== Trading Decorators Examples ===\n")
    
    print("1. Backtesting Integration Example:")
    await integrate_with_existing_backtester()
    print()
    
    print("2. Live Trading Integration Example:")
    await integrate_with_existing_live_trader()
    print()
    
    print("3. Model Operations Example:")
    model_ops = ModelOperationsWithDecorators()
    features = pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0],
        'feature_2': [0.5, 1.5, 2.5],
        'feature_3': [0.1, 0.2, 0.3]
    })
    
    model_weights = {
        'xgboost': 0.4,
        'lstm': 0.3,
        'random_forest': 0.3
    }
    
    result = await model_ops.predict_with_ensemble(features, model_weights)
    print(f"Ensemble prediction result: {result}")
    print()
    
    print("4. Market Data Operations Example:")
    market_data = MarketDataOperationsWithDecorators()
    
    data = await market_data.fetch_market_data('BTCUSDT', '1m', 100)
    print(f"Market data shape: {data.shape}")
    
    real_time_data = await market_data.fetch_real_time_data('BTCUSDT')
    print(f"Real-time data: {real_time_data}")
    print()
    
    print("=== Examples Completed ===")


if __name__ == "__main__":
    asyncio.run(main())