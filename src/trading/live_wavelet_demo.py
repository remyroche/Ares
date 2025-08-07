"""
Live Trading Wavelet Demo

This script demonstrates the computationally-aware wavelet integration
for live trading with performance monitoring and real-time signal generation.
"""

import asyncio
import time
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.trading.live_wavelet_integration import LiveWaveletIntegration
from src.utils.logger import system_logger


class LiveWaveletDemo:
    """
    Demo class for computationally-aware wavelet integration.
    
    Demonstrates:
    - Real-time signal generation
    - Performance monitoring
    - Integration with trading pipeline
    - Fallback mechanisms
    """
    
    def __init__(self, config_path: str = "src/config/live_wavelet_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = system_logger.getChild("LiveWaveletDemo")
        
        # Initialize wavelet integration
        self.wavelet_integration = LiveWaveletIntegration(self.config)
        
        # Demo state
        self.is_running = False
        self.demo_data = []
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize the demo."""
        try:
            self.logger.info("🚀 Initializing Live Wavelet Demo...")
            
            # Initialize wavelet integration
            success = await self.wavelet_integration.initialize()
            if not success:
                self.logger.error("Failed to initialize wavelet integration")
                return False
            
            # Generate demo data
            self._generate_demo_data()
            
            self.logger.info("✅ Live Wavelet Demo initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing demo: {e}")
            return False
    
    def _generate_demo_data(self) -> None:
        """Generate realistic demo market data."""
        try:
            # Generate 1000 data points of realistic price data
            np.random.seed(42)
            n_points = 1000
            
            # Base price with trend and volatility
            base_price = 50000
            trend = np.linspace(0, 0.1, n_points)  # 10% upward trend
            volatility = np.random.gamma(2, 0.02, n_points)
            
            # Generate prices with realistic movements
            returns = np.random.normal(0, volatility) + trend
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Add some market events (spikes, crashes, etc.)
            # Market crash at 300
            prices[300:320] *= 0.95
            # Recovery rally at 600
            prices[600:620] *= 1.05
            # Volatility spike at 800
            prices[800:850] += np.random.normal(0, 1000, 50)
            
            # Create OHLCV data
            self.demo_data = []
            for i in range(n_points):
                price = prices[i]
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = price * (1 + np.random.normal(0, 0.005))
                volume = np.random.uniform(1000, 10000)
                
                self.demo_data.append({
                    "timestamp": time.time() + i,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume
                })
            
            self.logger.info(f"📊 Generated {len(self.demo_data)} demo data points")
            
        except Exception as e:
            self.logger.error(f"Error generating demo data: {e}")
    
    async def run_demo(self, duration: int = 60) -> None:
        """
        Run the live wavelet demo.
        
        Args:
            duration: Demo duration in seconds
        """
        try:
            self.logger.info(f"🎬 Starting Live Wavelet Demo (duration: {duration}s)")
            self.is_running = True
            
            start_time = time.time()
            data_index = 0
            
            while self.is_running and (time.time() - start_time) < duration:
                # Get next data point
                if data_index >= len(self.demo_data):
                    data_index = 0  # Loop back to start
                
                market_data = self._create_market_data(data_index)
                
                # Process with wavelet integration
                results = await self.wavelet_integration.process_market_data(market_data)
                
                if results:
                    self._log_signal_results(results)
                
                # Update performance stats periodically
                if data_index % 50 == 0:
                    self._log_performance_stats()
                
                # Check health
                if data_index % 100 == 0:
                    self._check_health()
                
                data_index += 1
                await asyncio.sleep(0.1)  # 100ms intervals
            
            self.logger.info("✅ Live Wavelet Demo completed")
            self._log_final_stats()
            
        except Exception as e:
            self.logger.error(f"Error running demo: {e}")
        finally:
            self.is_running = False
    
    def _create_market_data(self, index: int) -> dict:
        """Create market data for demo."""
        try:
            if index >= len(self.demo_data):
                return {}
            
            data_point = self.demo_data[index]
            
            return {
                "price_data": pd.DataFrame([data_point]),
                "volume_data": pd.DataFrame({"volume": [data_point["volume"]]}),
                "timestamp": data_point["timestamp"],
                "symbol": "BTCUSDT",
                "exchange": "BINANCE"
            }
            
        except Exception as e:
            self.logger.error(f"Error creating market data: {e}")
            return {}
    
    def _log_signal_results(self, results: dict) -> None:
        """Log signal results."""
        try:
            signal = results.get("wavelet_signal", "hold")
            confidence = results.get("wavelet_confidence", 0.0)
            energy = results.get("wavelet_energy", 0.0)
            entropy = results.get("wavelet_entropy", 0.0)
            comp_time = results.get("wavelet_computation_time", 0.0)
            
            if signal != "hold":
                self.logger.info(f"📊 Signal: {signal.upper()} "
                               f"(confidence: {confidence:.2f}, "
                               f"energy: {energy:.4f}, "
                               f"entropy: {entropy:.4f}, "
                               f"time: {comp_time:.3f}s)")
            
        except Exception as e:
            self.logger.error(f"Error logging signal results: {e}")
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        try:
            stats = self.wavelet_integration.get_performance_stats()
            
            if stats:
                recent = stats.get("recent_signals", {})
                buy_count = recent.get("buy_count", 0)
                sell_count = recent.get("sell_count", 0)
                hold_count = recent.get("hold_count", 0)
                avg_confidence = recent.get("avg_confidence", 0.0)
                avg_time = recent.get("avg_computation_time", 0.0)
                
                self.logger.info(f"📈 Performance: "
                               f"Signals: {buy_count}B/{sell_count}S/{hold_count}H, "
                               f"Avg Confidence: {avg_confidence:.2f}, "
                               f"Avg Time: {avg_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error logging performance stats: {e}")
    
    def _check_health(self) -> None:
        """Check system health."""
        try:
            is_healthy = self.wavelet_integration.is_healthy()
            
            if not is_healthy:
                self.logger.warning("⚠️ Wavelet integration health check failed")
            else:
                self.logger.info("✅ Wavelet integration healthy")
                
        except Exception as e:
            self.logger.error(f"Error checking health: {e}")
    
    def _log_final_stats(self) -> None:
        """Log final statistics."""
        try:
            stats = self.wavelet_integration.get_performance_stats()
            
            self.logger.info("📊 Final Statistics:")
            self.logger.info(f"  Total signals: {stats.get('signal_history_count', 0)}")
            
            recent = stats.get("recent_signals", {})
            if recent:
                self.logger.info(f"  Buy signals: {recent.get('buy_count', 0)}")
                self.logger.info(f"  Sell signals: {recent.get('sell_count', 0)}")
                self.logger.info(f"  Hold signals: {recent.get('hold_count', 0)}")
                self.logger.info(f"  Average confidence: {recent.get('avg_confidence', 0.0):.2f}")
                self.logger.info(f"  Average computation time: {recent.get('avg_computation_time', 0.0):.3f}s")
            
            # Performance stats from analyzer
            perf_stats = stats.get("performance_stats", {})
            if perf_stats:
                self.logger.info(f"  Window size: {perf_stats.get('window_size', 0)}")
                self.logger.info(f"  Wavelet type: {perf_stats.get('wavelet_type', 'unknown')}")
                self.logger.info(f"  Signal rate: {perf_stats.get('signal_rate', 0.0):.2%}")
            
        except Exception as e:
            self.logger.error(f"Error logging final stats: {e}")
    
    def stop_demo(self) -> None:
        """Stop the demo."""
        self.is_running = False
        self.logger.info("🛑 Demo stopped by user")


async def main():
    """Main demo function."""
    try:
        # Create and initialize demo
        demo = LiveWaveletDemo()
        
        success = await demo.initialize()
        if not success:
            print("❌ Failed to initialize demo")
            return
        
        # Run demo for 60 seconds
        await demo.run_demo(duration=60)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error in demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())