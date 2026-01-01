"""
Live Trading Edge Preservation for Unified Layer2 Price System

This module implements strategies to maintain the edge gained from unified Kalman+VWAP
price generation when transitioning from backtesting to live trading.

Key Edge Preservation Strategies:
1. Parameter Stability Monitoring
2. Real-time Price Quality Validation
3. Adaptive Fallback Mechanisms
4. Performance Tracking & Alerts
5. Continuous Learning & Adaptation
"""

import logging
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LiveTradingMetrics:
    """Metrics for monitoring live trading edge preservation."""
    timestamp: datetime
    price_quality_score: float
    parameter_drift_score: float
    prediction_confidence: float
    market_regime_stability: float
    performance_slippage: float
    error_count: int

class EdgePreservationSystem:
    """
    Comprehensive edge preservation system for live trading.
    
    Monitors and maintains the performance advantage gained from
    unified Kalman+VWAP price generation in live environments.
    """
    
    def __init__(self, config_path: str = "config/live_trading_edge_config.json"):
        self.config_path = config_path
        self.metrics_history: List[LiveTradingMetrics] = []
        self.alert_thresholds = self._load_config()
        self.last_parameter_update = None
        self.edge_degradation_detected = False
        
    def _load_config(self) -> dict:
        """Load edge preservation configuration."""
        default_config = {
            'price_quality_threshold': 0.7,
            'parameter_drift_threshold': 0.15,
            'performance_slippage_threshold': 0.05,
            'min_samples_for_analysis': 100,
            'reoptimization_trigger_quality': 0.6,
            'alert_cooldown_minutes': 30,
            'max_error_rate': 0.02
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def monitor_live_performance(self, 
                              current_metrics: LiveTradingMetrics) -> Dict[str, any]:
        """
        Monitor live trading performance and detect edge degradation.
        
        Args:
            current_metrics: Current live trading metrics
            
        Returns:
            Analysis results with recommendations
        """
        self.metrics_history.append(current_metrics)
        
        # Analyze performance trends
        analysis = self._analyze_performance_trends()
        
        # Check for edge degradation
        degradation_detected = self._detect_edge_degradation(current_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis, degradation_detected)
        
        # Trigger alerts if needed
        if degradation_detected:
            self._trigger_edge_alert(current_metrics, analysis)
        
        return {
            'analysis': analysis,
            'degradation_detected': degradation_detected,
            'recommendations': recommendations,
            'metrics_count': len(self.metrics_history)
        }
    
    def _analyze_performance_trends(self) -> Dict[str, float]:
        """Analyze performance trends from metrics history."""
        if len(self.metrics_history) < self.alert_thresholds['min_samples_for_analysis']:
            return {'status': 'insufficient_data'}
        
        recent_metrics = self.metrics_history[-50:]  # Last 50 samples
        historical_metrics = self.metrics_history[:-50] if len(self.metrics_history) > 50 else []
        
        trends = {}
        
        # Price quality trend
        recent_quality = np.mean([m.price_quality_score for m in recent_metrics])
        if historical_metrics:
            historical_quality = np.mean([m.price_quality_score for m in historical_metrics])
            trends['price_quality_trend'] = recent_quality - historical_quality
        else:
            trends['price_quality_trend'] = 0.0
        
        # Parameter drift trend
        recent_drift = np.mean([m.parameter_drift_score for m in recent_metrics])
        if historical_metrics:
            historical_drift = np.mean([m.parameter_drift_score for m in historical_metrics])
            trends['parameter_drift_trend'] = recent_drift - historical_drift
        else:
            trends['parameter_drift_trend'] = 0.0
        
        # Performance slippage
        recent_slippage = np.mean([m.performance_slippage for m in recent_metrics])
        trends['performance_slippage'] = recent_slippage
        
        # Error rate
        recent_errors = np.mean([m.error_count for m in recent_metrics])
        trends['error_rate'] = recent_errors / len(recent_metrics)
        
        return trends
    
    def _detect_edge_degradation(self, current_metrics: LiveTradingMetrics) -> bool:
        """Detect if the trading edge is degrading."""
        degradation_signals = 0
        
        # Check price quality
        if current_metrics.price_quality_score < self.alert_thresholds['price_quality_threshold']:
            degradation_signals += 1
            logger.warning(f"Price quality degraded: {current_metrics.price_quality_score:.3f}")
        
        # Check parameter drift
        if current_metrics.parameter_drift_score > self.alert_thresholds['parameter_drift_threshold']:
            degradation_signals += 1
            logger.warning(f"Parameter drift detected: {current_metrics.parameter_drift_score:.3f}")
        
        # Check performance slippage
        if current_metrics.performance_slippage > self.alert_thresholds['performance_slippage_threshold']:
            degradation_signals += 1
            logger.warning(f"Performance slippage: {current_metrics.performance_slippage:.3f}")
        
        # Check error rate
        if len(self.metrics_history) > 20:
            recent_errors = np.mean([m.error_count for m in self.metrics_history[-20:]])
            if recent_errors > self.alert_thresholds['max_error_rate']:
                degradation_signals += 1
                logger.warning(f"High error rate: {recent_errors:.3f}")
        
        # Edge degradation detected if multiple signals
        edge_degraded = degradation_signals >= 2
        if edge_degraded and not self.edge_degradation_detected:
            self.edge_degradation_detected = True
            logger.error("EDGE DEGRADATION DETECTED - Immediate action required")
        
        return edge_degraded
    
    def _generate_recommendations(self, analysis: Dict[str, float], 
                                 degradation_detected: bool) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if analysis.get('status') == 'insufficient_data':
            recommendations.append("Collect more live data before making adjustments")
            return recommendations
        
        # Price quality recommendations
        if analysis.get('price_quality_trend', 0) < -0.1:
            recommendations.append("Consider re-optimizing Kalman parameters (Q, R)")
            recommendations.append("Check data quality and market regime changes")
        
        # Parameter drift recommendations
        if analysis.get('parameter_drift_trend', 0) > 0.1:
            recommendations.append("Schedule parameter re-optimization")
            recommendations.append("Consider adaptive parameter adjustment")
        
        # Performance slippage recommendations
        if analysis.get('performance_slippage', 0) > 0.03:
            recommendations.append("Review execution timing and slippage control")
            recommendations.append("Check market microstructure changes")
        
        # Error rate recommendations
        if analysis.get('error_rate', 0) > 0.01:
            recommendations.append("Investigate system stability and data feeds")
            recommendations.append("Review error handling and fallback mechanisms")
        
        # Edge degradation specific recommendations
        if degradation_detected:
            recommendations.extend([
                "IMMEDIATE: Switch to fallback price generation",
                "IMMEDIATE: Reduce position sizes until edge restored",
                "Schedule comprehensive parameter re-optimization",
                "Review market regime changes and adapt strategy"
            ])
        
        return recommendations
    
    def _trigger_edge_alert(self, current_metrics: LiveTradingMetrics, 
                           analysis: Dict[str, float]):
        """Trigger edge degradation alert."""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics.__dict__,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(analysis, True)
        }
        
        # Save alert to file
        alert_file = f"outcomes/edge_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2, default=str)
            logger.error(f"Edge alert saved to {alert_file}")
        except Exception as e:
            logger.error(f"Failed to save edge alert: {e}")
    
    def get_edge_status_report(self) -> Dict[str, any]:
        """Generate comprehensive edge status report."""
        if not self.metrics_history:
            return {'status': 'no_data', 'message': 'No metrics available'}
        
        recent_metrics = self.metrics_history[-10:]
        
        report = {
            'status': 'healthy' if not self.edge_degraded else 'degraded',
            'total_samples': len(self.metrics_history),
            'recent_performance': {
                'avg_price_quality': np.mean([m.price_quality_score for m in recent_metrics]),
                'avg_parameter_drift': np.mean([m.parameter_drift_score for m in recent_metrics]),
                'avg_performance_slippage': np.mean([m.performance_slippage for m in recent_metrics]),
                'avg_error_rate': np.mean([m.error_count for m in recent_metrics]) / 10
            },
            'trends': self._analyze_performance_trends(),
            'last_update': self.metrics_history[-1].timestamp.isoformat(),
            'edge_degraded': self.edge_degraded
        }
        
        return report

class AdaptiveParameterManager:
    """
    Manages adaptive parameter adjustment for live trading edge preservation.
    """
    
    def __init__(self, initial_params: Dict[str, float]):
        self.current_params = initial_params.copy()
        self.param_history = [initial_params.copy()]
        self.adjustment_history = []
        self.performance_window = 100
        
    def evaluate_parameter_performance(self, 
                                     performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate current parameter performance and suggest adjustments.
        
        Args:
            performance_metrics: Current performance metrics
            
        Returns:
            Parameter adjustment suggestions
        """
        suggestions = {}
        
        # Evaluate based on performance metrics
        if performance_metrics.get('price_quality_score', 1.0) < 0.7:
            # Suggest Kalman parameter adjustments
            suggestions['kalman_Q'] = self.current_params['kalman_Q'] * 1.2  # Increase responsiveness
            suggestions['kalman_R'] = self.current_params['kalman_R'] * 0.8  # Trust observations more
            suggestions['vwap_weight'] = max(0.1, self.current_params['vwap_weight'] - 0.1)  # Reduce VWAP weight
        
        if performance_metrics.get('parameter_drift_score', 0.0) > 0.15:
            # Suggest reversion to previous successful parameters
            if len(self.param_history) > 1:
                previous_params = self.param_history[-2]
                suggestions.update({
                    'kalman_Q': (self.current_params['kalman_Q'] + previous_params['kalman_Q']) / 2,
                    'kalman_R': (self.current_params['kalman_R'] + previous_params['kalman_R']) / 2,
                    'vwap_weight': (self.current_params['vwap_weight'] + previous_params['vwap_weight']) / 2
                })
        
        return suggestions
    
    def apply_parameter_adjustments(self, adjustments: Dict[str, float], 
                                   reason: str = "adaptive_adjustment"):
        """Apply parameter adjustments with tracking."""
        old_params = self.current_params.copy()
        
        # Apply adjustments with bounds checking
        for param, new_value in adjustments.items():
            if param in self.current_params:
                # Apply bounds
                if param == 'kalman_Q':
                    new_value = np.clip(new_value, 1e-6, 1e-2)
                elif param == 'kalman_R':
                    new_value = np.clip(new_value, 1e-4, 1e-1)
                elif param == 'vwap_weight':
                    new_value = np.clip(new_value, 0.0, 1.0)
                
                self.current_params[param] = new_value
        
        # Record adjustment
        self.adjustment_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_params': old_params,
            'new_params': self.current_params.copy(),
            'reason': reason
        })
        
        self.param_history.append(self.current_params.copy())
        
        logger.info(f"Parameter adjustment applied: {reason}")
        logger.info(f"New params: {self.current_params}")
    
    def get_parameter_stability_score(self) -> float:
        """Calculate parameter stability score (0-1, higher = more stable)."""
        if len(self.param_history) < 2:
            return 1.0
        
        # Calculate parameter variance
        param_arrays = {}
        for param in self.current_params.keys():
            param_arrays[param] = np.array([h[param] for h in self.param_history])
        
        # Calculate coefficient of variation for each parameter
        stability_scores = []
        for param, values in param_arrays.items():
            if len(values) > 1 and np.std(values) > 0:
                cv = np.std(values) / np.abs(np.mean(values))
                stability = 1 / (1 + cv)  # Convert to stability score
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 1.0

# Live Trading Integration Example
class LiveTradingLayer2Manager:
    """
    Integrates unified price generation with edge preservation in live trading.
    """
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol
        self.edge_system = EdgePreservationSystem()
        self.param_manager = None
        self.price_manager = None
        self.last_metrics = None
        
    def initialize_with_backtest_params(self, backtest_params: Dict[str, float]):
        """Initialize system with backtest-optimized parameters."""
        self.param_manager = AdaptiveParameterManager(backtest_params)
        self.price_manager = LiveTradingPriceManager(backtest_params)
        
        logger.info(f"Initialized live trading for {self.symbol}")
        logger.info(f"Backtest params: {backtest_params}")
    
    def generate_live_layer2_context(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Generate Layer2 context with edge preservation monitoring.
        
        Args:
            df: Current market data
            
        Returns:
            Tuple of (context DataFrame, edge monitoring results)
        """
        try:
            # Generate unified price with monitoring
            start_time = time.time()
            unified_price = self.price_manager.generate_live_price(df)
            price_generation_time = time.time() - start_time
            
            # Generate Layer2 context using unified price
            context = self._generate_layer2_features(df, unified_price)
            
            # Calculate current metrics
            current_metrics = self._calculate_current_metrics(
                df, unified_price, context, price_generation_time
            )
            
            # Monitor edge preservation
            edge_results = self.edge_system.monitor_live_performance(current_metrics)
            self.last_metrics = current_metrics
            
            # Check for adaptive parameter adjustments
            if edge_results['degradation_detected']:
                adjustments = self.param_manager.evaluate_parameter_performance(
                    edge_results['analysis']
                )
                if adjustments:
                    self.param_manager.apply_parameter_adjustments(
                        adjustments, "edge_degradation_response"
                    )
                    # Update price manager with new parameters
                    self.price_manager.layer0_params = self.param_manager.current_params
            
            return context, edge_results
            
        except Exception as e:
            logger.error(f"Live Layer2 generation failed: {e}")
            # Return fallback context
            return self._generate_fallback_context(df), {'status': 'error', 'error': str(e)}
    
    def _generate_layer2_features(self, df: pd.DataFrame, 
                                 unified_price: pd.Series) -> pd.DataFrame:
        """Generate Layer2 features using unified price."""
        from .unified_price_layer2 import UnifiedPriceMixin
        
        # Initialize generators with unified price
        vol_generator = UnifiedPriceMixin(use_unified_price=True)
        vol_generator._layer0_params = self.param_manager.current_params
        vol_generator._cached_unified_price = unified_price
        
        flow_generator = UnifiedPriceMixin(use_unified_price=True)
        flow_generator._layer0_params = self.param_manager.current_params
        flow_generator._cached_unified_price = unified_price
        
        trend_generator = UnifiedPriceMixin(use_unified_price=True)
        trend_generator._layer0_params = self.param_manager.current_params
        trend_generator._cached_unified_price = unified_price
        
        # Generate features
        volatility_context = vol_generator._get_unified_price(df)
        flow_metrics = flow_generator._get_unified_price(df)
        trend_context = trend_generator._get_unified_price(df)
        
        # Combine into context DataFrame
        context = pd.DataFrame({
            'unified_price': unified_price,
            'volatility_regime': volatility_context.pct_change().rolling(20).std(),
            'volume_pressure': flow_metrics.pct_change(),
            'trend_strength': trend_context.pct_change().rolling(10).mean()
        }, index=df.index)
        
        return context
    
    def _calculate_current_metrics(self, df: pd.DataFrame, 
                                 unified_price: pd.Series,
                                 context: pd.DataFrame,
                                 generation_time: float) -> LiveTradingMetrics:
        """Calculate current live trading metrics."""
        
        # Price quality score
        raw_price = df['close']
        tracking_error = np.mean((unified_price - raw_price) ** 2)
        max_acceptable_error = (raw_price.std() * 0.01) ** 2
        price_quality = 1 - min(tracking_error / max_acceptable_error, 1.0)
        
        # Parameter drift score
        param_stability = self.param_manager.get_parameter_stability_score()
        parameter_drift = 1 - param_stability
        
        # Prediction confidence (placeholder - would come from actual models)
        prediction_confidence = 0.75  # Would be calculated from model predictions
        
        # Market regime stability
        regime_volatility = context['volatility_regime'].rolling(50).std().iloc[-1]
        market_regime_stability = 1 - min(regime_volatility / 0.02, 1.0)  # Normalize
        
        # Performance slippage (placeholder - would come from actual trading)
        performance_slippage = 0.01  # Would be calculated from actual vs expected
        
        # Error count (placeholder - would track actual errors)
        error_count = 0
        
        return LiveTradingMetrics(
            timestamp=datetime.now(),
            price_quality_score=price_quality,
            parameter_drift_score=parameter_drift,
            prediction_confidence=prediction_confidence,
            market_regime_stability=market_regime_stability,
            performance_slippage=performance_slippage,
            error_count=error_count
        )
    
    def _generate_fallback_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fallback context when unified price fails."""
        logger.warning("Using fallback context generation")
        
        # Simple fallback using raw price
        return pd.DataFrame({
            'unified_price': df['close'],
            'volatility_regime': df['close'].pct_change().rolling(20).std(),
            'volume_pressure': df.get('volume', pd.Series(0, index=df.index)).pct_change(),
            'trend_strength': df['close'].pct_change().rolling(10).mean()
        }, index=df.index)
    
    def get_edge_status(self) -> Dict[str, any]:
        """Get current edge preservation status."""
        return self.edge_system.get_edge_status_report()
    
    def save_edge_report(self, filepath: str = None):
        """Save comprehensive edge preservation report."""
        if filepath is None:
            filepath = f"outcomes/edge_report_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'edge_status': self.get_edge_status(),
            'parameter_history': self.param_manager.param_history if self.param_manager else [],
            'adjustment_history': self.param_manager.adjustment_history if self.param_manager else [],
            'current_params': self.param_manager.current_params if self.param_manager else {},
            'last_metrics': self.last_metrics.__dict__ if self.last_metrics else None
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Edge report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save edge report: {e}")

# Usage Example for Live Trading
def setup_live_trading_edge_system(symbol: str, backtest_params: Dict[str, float]):
    """
    Setup complete edge preservation system for live trading.
    
    Args:
        symbol: Trading symbol
        backtest_params: Parameters optimized in backtesting
    """
    # Initialize live trading manager
    live_manager = LiveTradingLayer2Manager(symbol)
    live_manager.initialize_with_backtest_params(backtest_params)
    
    logger.info(f"Live trading edge system setup complete for {symbol}")
    logger.info(f"Edge preservation monitoring active")
    
    return live_manager
