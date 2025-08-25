#!/usr/bin/env python3
"""
Simple Test Script for Per-HMM Regime Triple Barrier Thresholds and TPSL Parameters Optimization

This script demonstrates the core functionality of the per-HMM regime optimization system
without depending on the existing codebase infrastructure.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit


class SimplePerHMMRegimeTPSLOptimizer:
    """
    Simplified version of the per-HMM regime TPSL optimizer for demonstration purposes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = self._create_simple_logger()
        
        # Optimization configuration
        self.optimization_config = config.get("per_hmm_regime_tpsl_optimizer", {})
        self.n_trials = self.optimization_config.get("n_trials", 50)  # Reduced for demo
        self.min_trades_per_regime = self.optimization_config.get("min_trades_per_regime", 10)
        self.cv_folds = self.optimization_config.get("cv_folds", 3)
        self.optimization_metric = self.optimization_config.get("optimization_metric", "sharpe_ratio")
        
        # Regime-specific default parameters
        self.regime_defaults = {
            "hmm_cluster_0": {
                "name": "Low Volatility Sideways",
                "triple_barrier": {"profit_take_multiplier": 0.003, "stop_loss_multiplier": 0.002, "time_barrier_minutes": 45},
                "tpsl": {"target_pct": 0.005, "stop_pct": 0.003, "risk_reward_ratio": 1.67},
                "characteristics": {"volatility": "low", "trend": "sideways", "frequency": "high"}
            },
            "hmm_cluster_1": {
                "name": "Moderate Volatility Trending",
                "triple_barrier": {"profit_take_multiplier": 0.005, "stop_loss_multiplier": 0.003, "time_barrier_minutes": 60},
                "tpsl": {"target_pct": 0.008, "stop_pct": 0.004, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "moderate", "trend": "trending", "frequency": "medium"}
            },
            "hmm_cluster_2": {
                "name": "High Volatility Breakout",
                "triple_barrier": {"profit_take_multiplier": 0.008, "stop_loss_multiplier": 0.004, "time_barrier_minutes": 30},
                "tpsl": {"target_pct": 0.012, "stop_pct": 0.006, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "high", "trend": "breakout", "frequency": "low"}
            },
            "hmm_cluster_3": {
                "name": "Extreme Volatility Crisis",
                "triple_barrier": {"profit_take_multiplier": 0.015, "stop_loss_multiplier": 0.008, "time_barrier_minutes": 20},
                "tpsl": {"target_pct": 0.02, "stop_pct": 0.01, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "extreme", "trend": "crisis", "frequency": "very_low"}
            }
        }
        
        # Optimization results cache
        self.optimization_results: Dict[str, Dict[str, Any]] = {}
        
    def _create_simple_logger(self):
        """Create a simple logger for demonstration."""
        class SimpleLogger:
            def info(self, msg):
                print(f"ℹ️  {msg}")
            def warning(self, msg):
                print(f"⚠️  {msg}")
            def error(self, msg):
                print(f"❌ {msg}")
            def exception(self, msg):
                print(f"💥 {msg}")
        return SimpleLogger()
    
    def generate_mock_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate mock OHLCV data for testing.
        
        Args:
            symbol: Symbol name
            days: Number of days of data to generate
            
        Returns:
            pd.DataFrame: Mock OHLCV data
        """
        try:
            self.logger.info(f"📊 Generating {days} days of mock data for {symbol}...")
            
            # Generate timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            timestamps = pd.date_range(start=start_time, end=end_time, freq='30min')
            
            # Generate price data with realistic patterns
            np.random.seed(42)  # For reproducible results
            
            # Base price
            base_price = 2000.0 if 'ETH' in symbol else 50000.0
            
            # Generate price movements
            returns = np.random.normal(0, 0.002, len(timestamps))  # 0.2% daily volatility
            
            # Add some regime-like patterns
            for i in range(0, len(returns), 48):  # Every 24 hours (48 * 30min)
                if i + 48 < len(returns):
                    # Add trend or mean reversion patterns
                    pattern = np.random.choice(['trend', 'mean_reversion', 'volatile'])
                    if pattern == 'trend':
                        returns[i:i+48] += np.linspace(0, 0.01, 48)  # Upward trend
                    elif pattern == 'mean_reversion':
                        returns[i:i+48] = returns[i:i+48] * 0.5  # Reduced volatility
                    else:  # volatile
                        returns[i:i+48] *= 2.0  # Increased volatility
            
            # Calculate prices
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate OHLCV data
            data = []
            for i, (ts, price) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC from close price
                volatility = abs(returns[i]) if i < len(returns) else 0.001
                
                high = price * (1 + np.random.uniform(0, volatility * 2))
                low = price * (1 - np.random.uniform(0, volatility * 2))
                open_price = price * (1 + np.random.uniform(-volatility, volatility))
                
                # Ensure OHLC consistency
                high = max(high, open_price, price)
                low = min(low, open_price, price)
                
                # Generate volume
                volume = np.random.uniform(1000, 10000)
                
                data.append({
                    'timestamp': ts,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"✅ Generated {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error generating mock data: {e}")
            return pd.DataFrame()
    
    def simulate_triple_barrier_labels(self, data: pd.DataFrame, tb_params: Dict[str, Any]) -> pd.Series:
        """Simulate triple barrier labeling.
        
        Args:
            data: OHLCV data
            tb_params: Triple barrier parameters
            
        Returns:
            pd.Series: Labels (1 for buy, -1 for sell, 0 for hold)
        """
        try:
            labels = pd.Series(0, index=data.index)
            
            for i in range(len(data) - 1):
                current_price = data.iloc[i]['close']
                profit_barrier = current_price * (1 + tb_params['profit_take_multiplier'])
                stop_barrier = current_price * (1 - tb_params['stop_loss_multiplier'])
                
                # Look ahead for barrier hits
                lookahead = min(tb_params.get('max_lookahead', 100), len(data) - i - 1)
                
                for j in range(i + 1, min(i + 1 + lookahead, len(data))):
                    high = data.iloc[j]['high']
                    low = data.iloc[j]['low']
                    
                    if high >= profit_barrier:
                        labels.iloc[i] = 1  # Buy signal
                        break
                    elif low <= stop_barrier:
                        labels.iloc[i] = -1  # Sell signal
                        break
            
            return labels
            
        except Exception as e:
            self.logger.error(f"❌ Error in triple barrier labeling: {e}")
            return pd.Series(0, index=data.index)
    
    def simulate_trades(self, data: pd.DataFrame, labels: pd.Series, tpsl_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate trades using TPSL parameters.
        
        Args:
            data: OHLCV data
            labels: Triple barrier labels
            tpsl_params: TPSL parameters
            
        Returns:
            List[Dict[str, Any]]: List of trade dictionaries
        """
        trades = []
        position_open = False
        entry_price = 0.0
        entry_time = None
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            
            if not position_open:
                # Entry condition based on triple barrier labels
                if labels.iloc[i-1] == 1:  # Buy signal
                    position_open = True
                    entry_price = current_price
                    entry_time = data.index[i-1]
                elif labels.iloc[i-1] == -1:  # Sell signal
                    position_open = True
                    entry_price = current_price
                    entry_time = data.index[i-1]
            else:
                # Check for TP/SL
                if high_price >= entry_price * (1 + tpsl_params['target_pct']):
                    # Take profit hit
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": data.index[i],
                        "entry_price": entry_price,
                        "exit_price": entry_price * (1 + tpsl_params['target_pct']),
                        "return": tpsl_params['target_pct'],
                        "type": "TP"
                    })
                    position_open = False
                elif low_price <= entry_price * (1 - tpsl_params['stop_pct']):
                    # Stop loss hit
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": data.index[i],
                        "entry_price": entry_price,
                        "exit_price": entry_price * (1 - tpsl_params['stop_pct']),
                        "return": -tpsl_params['stop_pct'],
                        "type": "SL"
                    })
                    position_open = False
        
        return trades
    
    def calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not trades:
            return {"sharpe_ratio": -1.0, "total_return": 0.0, "win_rate": 0.0, "max_drawdown": 1.0}
        
        returns = [trade["return"] for trade in trades]
        total_return = sum(returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # Calculate max drawdown
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 1.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "num_trades": len(trades)
        }
    
    def optimize_regime_parameters(self, regime: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize parameters for a specific regime.
        
        Args:
            regime: Regime to optimize
            historical_data: Historical data for optimization
            
        Returns:
            Dict[str, Any]: Optimized parameters
        """
        try:
            self.logger.info(f"🎯 Optimizing parameters for regime: {regime}")
            
            # Get base parameters for this regime
            base_params = self.regime_defaults.get(regime, self.regime_defaults["hmm_cluster_0"])
            
            # Create optimization study
            study = optuna.create_study(
                direction="maximize",
                study_name=f"per_hmm_tpsl_optimization_{regime}",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Define objective function
            def objective(trial):
                # Suggest triple barrier parameters
                tb_params = {
                    "profit_take_multiplier": trial.suggest_float("tb_profit_take", 0.001, 0.01),
                    "stop_loss_multiplier": trial.suggest_float("tb_stop_loss", 0.0005, 0.005),
                    "time_barrier_minutes": trial.suggest_int("tb_time_barrier", 15, 120),
                    "max_lookahead": trial.suggest_int("tb_max_lookahead", 50, 200)
                }
                
                # Suggest TPSL parameters
                tpsl_params = {
                    "target_pct": trial.suggest_float("tpsl_target", 0.002, 0.02),
                    "stop_pct": trial.suggest_float("tpsl_stop", 0.001, 0.01),
                    "risk_reward_ratio": trial.suggest_float("tpsl_rr", 1.5, 4.0)
                }
                
                # Validate parameter constraints
                if tpsl_params["target_pct"] <= tpsl_params["stop_pct"]:
                    return -1.0
                
                if tb_params["profit_take_multiplier"] <= tb_params["stop_loss_multiplier"]:
                    return -1.0
                
                # Run cross-validation
                cv_scores = []
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                
                for train_idx, test_idx in tscv.split(historical_data):
                    test_data = historical_data.iloc[test_idx]
                    
                    # Generate labels
                    labels = self.simulate_triple_barrier_labels(test_data, tb_params)
                    
                    # Simulate trades
                    trades = self.simulate_trades(test_data, labels, tpsl_params)
                    
                    if len(trades) < self.min_trades_per_regime:
                        cv_scores.append(-1.0)
                        continue
                    
                    # Calculate performance
                    metrics = self.calculate_performance_metrics(trades)
                    
                    if self.optimization_metric == "sharpe_ratio":
                        score = metrics["sharpe_ratio"]
                    elif self.optimization_metric == "total_return":
                        score = metrics["total_return"]
                    elif self.optimization_metric == "win_rate":
                        score = metrics["win_rate"]
                    else:
                        score = metrics["sharpe_ratio"]
                    
                    cv_scores.append(score)
                
                # Return mean CV score
                mean_score = np.mean(cv_scores) if cv_scores else -1.0
                return mean_score
            
            # Run optimization
            self.logger.info(f"🔄 Running optimization for {regime} with {self.n_trials} trials...")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Combine with base parameters
            optimized_params = {
                **base_params,
                "optimized_triple_barrier": {
                    "profit_take_multiplier": best_params.get("tb_profit_take", 0.003),
                    "stop_loss_multiplier": best_params.get("tb_stop_loss", 0.002),
                    "time_barrier_minutes": best_params.get("tb_time_barrier", 45),
                    "max_lookahead": best_params.get("tb_max_lookahead", 100)
                },
                "optimized_tpsl": {
                    "target_pct": best_params.get("tpsl_target", 0.005),
                    "stop_pct": best_params.get("tpsl_stop", 0.003),
                    "risk_reward_ratio": best_params.get("tpsl_rr", 1.67)
                },
                "optimization_score": best_value,
                "optimization_trials": self.n_trials,
                "optimization_time": datetime.now().isoformat()
            }
            
            # Cache results
            self.optimization_results[regime] = optimized_params
            
            self.logger.info(f"✅ Optimized parameters for {regime}: score={best_value:.4f}")
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing parameters for regime {regime}: {e}")
            return self.regime_defaults.get(regime, self.regime_defaults["hmm_cluster_0"])
    
    def get_optimized_parameters(self, current_data: pd.DataFrame, historical_data: pd.DataFrame, 
                                exchange: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get optimized parameters for the current market conditions.
        
        Args:
            current_data: Current market data
            historical_data: Historical data for optimization
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe string
            
        Returns:
            Dict[str, Any]: Optimized parameters
        """
        try:
            # For demonstration, we'll optimize for a few regimes
            regimes_to_optimize = ["hmm_cluster_0", "hmm_cluster_1", "hmm_cluster_2"]
            
            best_regime = None
            best_score = -1.0
            
            for regime in regimes_to_optimize:
                optimized_params = self.optimize_regime_parameters(regime, historical_data)
                score = optimized_params.get("optimization_score", -1.0)
                
                if score > best_score:
                    best_score = score
                    best_regime = regime
            
            if best_regime:
                return {
                    **self.optimization_results[best_regime],
                    "selected_regime": best_regime,
                    "confidence": 0.8,  # Mock confidence
                    "source": "optimized"
                }
            else:
                return {
                    **self.regime_defaults["hmm_cluster_0"],
                    "selected_regime": "hmm_cluster_0",
                    "confidence": 0.5,
                    "source": "fallback"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting optimized parameters: {e}")
            return {
                **self.regime_defaults["hmm_cluster_0"],
                "selected_regime": "hmm_cluster_0",
                "confidence": 0.5,
                "source": "fallback"
            }
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime optimization.
        
        Returns:
            Dict[str, Any]: Optimization statistics
        """
        return {
            "optimized_regimes": list(self.optimization_results.keys()),
            "total_optimizations": len(self.optimization_results),
            "regime_scores": {
                regime: params.get("optimization_score", -1.0)
                for regime, params in self.optimization_results.items()
            }
        }


async def main():
    """Main function to demonstrate the system."""
    print("🚀 PER-HMM REGIME TPSL OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create configuration
    config = {
        "per_hmm_regime_tpsl_optimizer": {
            "n_trials": 20,  # Reduced for demo
            "min_trades_per_regime": 5,  # Reduced for demo
            "cv_folds": 2,  # Reduced for demo
            "optimization_metric": "sharpe_ratio"
        }
    }
    
    # Create optimizer
    optimizer = SimplePerHMMRegimeTPSLOptimizer(config)
    
    # Generate mock data
    print("\n📊 Generating mock market data...")
    historical_data = optimizer.generate_mock_data("ETHUSDT", days=30)
    current_data = optimizer.generate_mock_data("ETHUSDT", days=7)
    
    if historical_data.empty or current_data.empty:
        print("❌ Failed to generate data")
        return
    
    # Test optimization
    print("\n🎯 Testing per-HMM regime TPSL optimization...")
    optimized_params = optimizer.get_optimized_parameters(
        current_data, historical_data, "BINANCE", "ETHUSDT", "30m"
    )
    
    # Display results
    print("\n📊 OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Selected Regime: {optimized_params.get('selected_regime', 'unknown')}")
    print(f"Confidence: {optimized_params.get('confidence', 0):.3f}")
    print(f"Source: {optimized_params.get('source', 'unknown')}")
    print(f"Optimization Score: {optimized_params.get('optimization_score', -1):.4f}")
    
    # Display triple barrier parameters
    tb_params = optimized_params.get('optimized_triple_barrier', {})
    if tb_params:
        print(f"\n🎯 Triple Barrier Parameters:")
        print(f"  Profit Take Multiplier: {tb_params.get('profit_take_multiplier', 0):.6f}")
        print(f"  Stop Loss Multiplier: {tb_params.get('stop_loss_multiplier', 0):.6f}")
        print(f"  Time Barrier Minutes: {tb_params.get('time_barrier_minutes', 0)}")
        print(f"  Max Lookahead: {tb_params.get('max_lookahead', 0)}")
    
    # Display TPSL parameters
    tpsl_params = optimized_params.get('optimized_tpsl', {})
    if tpsl_params:
        print(f"\n💰 TPSL Parameters:")
        print(f"  Target %: {tpsl_params.get('target_pct', 0):.4f}")
        print(f"  Stop %: {tpsl_params.get('stop_pct', 0):.4f}")
        print(f"  Risk-Reward Ratio: {tpsl_params.get('risk_reward_ratio', 0):.2f}")
    
    # Display regime characteristics
    characteristics = optimized_params.get('characteristics', {})
    if characteristics:
        print(f"\n📈 Regime Characteristics:")
        print(f"  Volatility: {characteristics.get('volatility', 'unknown')}")
        print(f"  Trend: {characteristics.get('trend', 'unknown')}")
        print(f"  Frequency: {characteristics.get('frequency', 'unknown')}")
    
    # Display statistics
    print(f"\n📊 OPTIMIZATION STATISTICS")
    print("=" * 60)
    stats = optimizer.get_regime_statistics()
    print(f"Optimized Regimes: {len(stats.get('optimized_regimes', []))}")
    print(f"Total Optimizations: {stats.get('total_optimizations', 0)}")
    
    regime_scores = stats.get('regime_scores', {})
    if regime_scores:
        print(f"\nRegime Scores:")
        for regime, score in regime_scores.items():
            print(f"  {regime}: {score:.4f}")
    
    print(f"\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())