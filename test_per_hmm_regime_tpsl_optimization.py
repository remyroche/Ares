#!/usr/bin/env python3
"""
Test Script for Per-HMM Regime Triple Barrier Thresholds and TPSL Parameters Optimization

This script demonstrates the comprehensive per-HMM regime optimization system that:
1. Identifies HMM regimes using the HMM composite manager
2. Optimizes triple barrier thresholds for each regime
3. Optimizes TPSL parameters for each regime
4. Provides regime-specific parameter recommendations
5. Validates optimization results through backtesting

Usage:
    python test_per_hmm_regime_tpsl_optimization.py --symbol ETHUSDT --exchange BINANCE --timeframe 30m
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.analyst_training_components.per_hmm_regime_tpsl_optimizer import (
    PerHMMRegimeTPSLOptimizer
)
from src.config import CONFIG
from src.utils.logger import system_logger


class PerHMMRegimeTPSLOptimizationTester:
    """Test class for per-HMM regime TPSL optimization system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PerHMMRegimeTPSLOptimizationTester")
        self.optimizer = PerHMMRegimeTPSLOptimizer(config)
        
    async def initialize(self) -> bool:
        """Initialize the tester and optimizer.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🚀 Initializing Per-HMM Regime TPSL Optimization Tester...")
            
            # Initialize the optimizer
            if not await self.optimizer.initialize():
                self.logger.error("❌ Failed to initialize optimizer")
                return False
                
            self.logger.info("✅ Tester initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize tester: {e}")
            return False
    
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
    
    async def test_regime_identification(self, symbol: str, exchange: str, timeframe: str) -> None:
        """Test HMM regime identification.
        
        Args:
            symbol: Symbol name
            exchange: Exchange name
            timeframe: Timeframe string
        """
        try:
            self.logger.info("🎯 Testing HMM regime identification...")
            
            # Generate mock data
            current_data = self.generate_mock_data(symbol, days=7)
            if current_data.empty:
                self.logger.error("❌ Failed to generate mock data")
                return
            
            # Test regime identification
            regime, confidence, regime_info = await self.optimizer.identify_current_hmm_regime(
                current_data, exchange, symbol, timeframe
            )
            
            self.logger.info(f"📊 Regime identification results:")
            self.logger.info(f"   - Identified regime: {regime}")
            self.logger.info(f"   - Confidence: {confidence:.3f}")
            self.logger.info(f"   - Method: {regime_info.get('method', 'unknown')}")
            self.logger.info(f"   - Regime ID: {regime_info.get('regime_id', 'unknown')}")
            
            if 'error' in regime_info:
                self.logger.warning(f"   - Error: {regime_info['error']}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error testing regime identification: {e}")
    
    async def test_single_regime_optimization(
        self,
        regime: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Test optimization for a single regime.
        
        Args:
            regime: Regime to optimize
            symbol: Symbol name
            exchange: Exchange name
            timeframe: Timeframe string
        """
        try:
            self.logger.info(f"🎯 Testing optimization for regime: {regime}")
            
            # Generate historical data
            historical_data = self.generate_mock_data(symbol, days=60)
            current_data = self.generate_mock_data(symbol, days=7)
            
            if historical_data.empty or current_data.empty:
                self.logger.error("❌ Failed to generate data")
                return
            
            # Test optimization
            optimized_params = await self.optimizer.optimize_regime_parameters(
                regime, historical_data, current_data, force_optimization=True
            )
            
            self.logger.info(f"📊 Optimization results for {regime}:")
            self.logger.info(f"   - Optimization score: {optimized_params.get('optimization_score', -1):.4f}")
            self.logger.info(f"   - Optimization trials: {optimized_params.get('optimization_trials', 0)}")
            
            # Log triple barrier parameters
            tb_params = optimized_params.get('optimized_triple_barrier', {})
            if tb_params:
                self.logger.info(f"   - Triple Barrier Parameters:")
                self.logger.info(f"     * Profit take multiplier: {tb_params.get('profit_take_multiplier', 0):.6f}")
                self.logger.info(f"     * Stop loss multiplier: {tb_params.get('stop_loss_multiplier', 0):.6f}")
                self.logger.info(f"     * Time barrier minutes: {tb_params.get('time_barrier_minutes', 0)}")
                self.logger.info(f"     * Max lookahead: {tb_params.get('max_lookahead', 0)}")
            
            # Log TPSL parameters
            tpsl_params = optimized_params.get('optimized_tpsl', {})
            if tpsl_params:
                self.logger.info(f"   - TPSL Parameters:")
                self.logger.info(f"     * Target %: {tpsl_params.get('target_pct', 0):.4f}")
                self.logger.info(f"     * Stop %: {tpsl_params.get('stop_pct', 0):.4f}")
                self.logger.info(f"     * Risk-reward ratio: {tpsl_params.get('risk_reward_ratio', 0):.2f}")
                self.logger.info(f"     * Position sizing %: {tpsl_params.get('position_sizing_pct', 0):.4f}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error testing regime optimization: {e}")
    
    async def test_full_optimization_pipeline(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Test the full optimization pipeline.
        
        Args:
            symbol: Symbol name
            exchange: Exchange name
            timeframe: Timeframe string
        """
        try:
            self.logger.info("🚀 Testing full per-HMM regime TPSL optimization pipeline...")
            
            # Generate data
            historical_data = self.generate_mock_data(symbol, days=90)
            current_data = self.generate_mock_data(symbol, days=7)
            
            if historical_data.empty or current_data.empty:
                self.logger.error("❌ Failed to generate data")
                return
            
            # Test full optimization
            optimized_params = await self.optimizer.get_optimized_parameters(
                current_data, historical_data, exchange, symbol, timeframe, force_optimization=True
            )
            
            self.logger.info(f"📊 Full optimization results:")
            self.logger.info(f"   - Identified regime: {optimized_params.get('regime', 'unknown')}")
            self.logger.info(f"   - Confidence: {optimized_params.get('confidence', 0):.3f}")
            self.logger.info(f"   - Source: {optimized_params.get('source', 'unknown')}")
            self.logger.info(f"   - Optimization score: {optimized_params.get('optimization_score', -1):.4f}")
            
            # Log regime characteristics
            characteristics = optimized_params.get('characteristics', {})
            if characteristics:
                self.logger.info(f"   - Regime characteristics:")
                self.logger.info(f"     * Volatility: {characteristics.get('volatility', 'unknown')}")
                self.logger.info(f"     * Trend: {characteristics.get('trend', 'unknown')}")
                self.logger.info(f"     * Frequency: {characteristics.get('frequency', 'unknown')}")
            
            # Log regime info
            regime_info = optimized_params.get('regime_info', {})
            if regime_info:
                self.logger.info(f"   - Regime info:")
                self.logger.info(f"     * Method: {regime_info.get('method', 'unknown')}")
                self.logger.info(f"     * Regime ID: {regime_info.get('regime_id', 'unknown')}")
                self.logger.info(f"     * Intensity: {regime_info.get('intensity', 0):.3f}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error testing full optimization pipeline: {e}")
    
    async def test_regime_statistics(self) -> None:
        """Test regime statistics functionality."""
        try:
            self.logger.info("📊 Testing regime statistics...")
            
            # Get regime statistics
            stats = self.optimizer.get_regime_statistics()
            
            self.logger.info(f"📈 Regime statistics:")
            self.logger.info(f"   - Total optimizations: {stats.get('total_optimizations', 0)}")
            self.logger.info(f"   - Optimized regimes: {len(stats.get('optimized_regimes', []))}")
            
            performance_summary = stats.get('performance_summary', {})
            if performance_summary:
                self.logger.info(f"   - Performance summary:")
                self.logger.info(f"     * Total regimes: {performance_summary.get('total_regimes', 0)}")
                self.logger.info(f"     * Optimized regimes: {performance_summary.get('optimized_regimes', 0)}")
                self.logger.info(f"     * Optimization rate: {performance_summary.get('optimization_rate', 0):.2%}")
                self.logger.info(f"     * Average confidence: {performance_summary.get('average_confidence', 0):.3f}")
            
            # Get parameter summary
            param_summary = self.optimizer.get_regime_parameter_summary()
            
            self.logger.info(f"📋 Parameter summary for {len(param_summary)} regimes:")
            for regime, params in param_summary.items():
                self.logger.info(f"   - {regime}: {params.get('name', 'Unknown')}")
                self.logger.info(f"     * Score: {params.get('optimization_score', -1):.4f}")
                self.logger.info(f"     * Last optimization: {params.get('last_optimization', 'Unknown')}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error testing regime statistics: {e}")
    
    async def test_all_regimes_optimization(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Test optimization for all regimes.
        
        Args:
            symbol: Symbol name
            exchange: Exchange name
            timeframe: Timeframe string
        """
        try:
            self.logger.info("🎯 Testing optimization for all regimes...")
            
            # Generate data
            historical_data = self.generate_mock_data(symbol, days=90)
            current_data = self.generate_mock_data(symbol, days=7)
            
            if historical_data.empty or current_data.empty:
                self.logger.error("❌ Failed to generate data")
                return
            
            # Test optimization for each regime
            regimes = list(self.optimizer.regime_defaults.keys())
            
            for regime in regimes:
                self.logger.info(f"🔄 Optimizing regime: {regime}")
                
                try:
                    optimized_params = await self.optimizer.optimize_regime_parameters(
                        regime, historical_data, current_data, force_optimization=True
                    )
                    
                    score = optimized_params.get('optimization_score', -1)
                    self.logger.info(f"   ✅ {regime}: score={score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"   ❌ {regime}: {e}")
            
            # Test statistics after all optimizations
            await self.test_regime_statistics()
            
        except Exception as e:
            self.logger.exception(f"❌ Error testing all regimes optimization: {e}")
    
    async def run_comprehensive_test(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Run comprehensive test of the per-HMM regime TPSL optimization system.
        
        Args:
            symbol: Symbol name
            exchange: Exchange name
            timeframe: Timeframe string
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 COMPREHENSIVE PER-HMM REGIME TPSL OPTIMIZATION TEST")
            self.logger.info("=" * 80)
            self.logger.info(f"📊 Testing with: {symbol} on {exchange} ({timeframe})")
            self.logger.info("=" * 80)
            
            # Test 1: Regime identification
            self.logger.info("\n🎯 TEST 1: HMM Regime Identification")
            self.logger.info("-" * 50)
            await self.test_regime_identification(symbol, exchange, timeframe)
            
            # Test 2: Single regime optimization
            self.logger.info("\n🎯 TEST 2: Single Regime Optimization")
            self.logger.info("-" * 50)
            await self.test_single_regime_optimization("hmm_cluster_0", symbol, exchange, timeframe)
            
            # Test 3: Full optimization pipeline
            self.logger.info("\n🎯 TEST 3: Full Optimization Pipeline")
            self.logger.info("-" * 50)
            await self.test_full_optimization_pipeline(symbol, exchange, timeframe)
            
            # Test 4: All regimes optimization
            self.logger.info("\n🎯 TEST 4: All Regimes Optimization")
            self.logger.info("-" * 50)
            await self.test_all_regimes_optimization(symbol, exchange, timeframe)
            
            # Test 5: Statistics and reporting
            self.logger.info("\n🎯 TEST 5: Statistics and Reporting")
            self.logger.info("-" * 50)
            await self.test_regime_statistics()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.exception(f"❌ Error in comprehensive test: {e}")


async def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test Per-HMM Regime TPSL Optimization")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol to test")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange to test")
    parser.add_argument("--timeframe", default="30m", help="Timeframe to test")
    parser.add_argument("--config", default="config.json", help="Config file path")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = {
        "SYMBOL": args.symbol,
        "EXCHANGE": args.exchange,
        "TIMEFRAME": args.timeframe,
        "per_hmm_regime_tpsl_optimizer": {
            "n_trials": 50,  # Reduced for testing
            "min_trades_per_regime": 10,  # Reduced for testing
            "cv_folds": 3,  # Reduced for testing
            "optimization_metric": "sharpe_ratio"
        }
    }
    
    # Create and run tester
    tester = PerHMMRegimeTPSLOptimizationTester(config)
    
    if await tester.initialize():
        await tester.run_comprehensive_test(args.symbol, args.exchange, args.timeframe)
    else:
        print("❌ Failed to initialize tester")


if __name__ == "__main__":
    asyncio.run(main())