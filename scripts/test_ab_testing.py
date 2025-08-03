#!/usr/bin/env python3
"""
Test script for Exchange A/B Testing Framework

This script demonstrates how to use the A/B testing framework to compare
model performance across different exchanges.
"""

import asyncio
import sys
import os
from datetime import datetime
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.supervisor.exchange_ab_tester import setup_exchange_ab_tester, ABTestConfig
from src.utils.logger import system_logger


class MockModel:
    """Mock model for testing purposes."""
    
    def __init__(self):
        self.confidence_score = 0.0
    
    def predict(self, features):
        """Mock prediction method."""
        # Simulate realistic predictions
        base_prediction = np.random.normal(0.01, 0.02)  # Small positive bias
        return base_prediction
    
    def get_confidence(self):
        """Mock confidence score."""
        # Simulate confidence based on prediction strength
        return min(0.95, max(0.3, abs(self.predict([])) * 10))


class MockMarketData:
    """Mock market data generator."""
    
    def __init__(self):
        self.base_price = 3500.0
        self.base_volume = 1000000
    
    def get_market_data(self, exchange: str):
        """Generate mock market data for an exchange."""
        # Exchange-specific adjustments
        volume_multipliers = {
            "BINANCE": 1.0,
            "MEXC": 0.05,  # 5% of Binance volume
            "GATEIO": 0.03  # 3% of Binance volume
        }
        
        spread_multipliers = {
            "BINANCE": 1.0,
            "MEXC": 2.5,
            "GATEIO": 3.0
        }
        
        # Generate realistic market data
        volume = self.base_volume * volume_multipliers.get(exchange.upper(), 1.0)
        spread = 0.001 * spread_multipliers.get(exchange.upper(), 1.0)
        price = self.base_price + np.random.normal(0, 10)
        
        return {
            "price": price,
            "volume": volume,
            "spread": spread,
            "timestamp": datetime.now()
        }


async def run_ab_test_demo():
    """Run a demonstration A/B test."""
    
    # Initialize logger
    logger = system_logger.getChild("ABTestDemo")
    logger.info("🚀 Starting A/B Testing Demo")
    
    # Configuration
    config = {
        "exchange_ab_tester": {
            "result_storage_path": "ab_test_results"
        }
    }
    
    try:
        # Initialize A/B tester
        logger.info("📊 Initializing A/B tester...")
        ab_tester = await setup_exchange_ab_tester(config)
        
        if not ab_tester:
            logger.error("❌ Failed to initialize A/B tester")
            return
        
        # Configure test
        test_config = ABTestConfig(
            test_name="demo_eth_comparison",
            model_id="demo_lstm_v1",
            exchanges=["BINANCE", "MEXC", "GATEIO"],
            test_duration_hours=1,  # Short demo
            min_confidence_threshold=0.6,
            max_position_size=0.05
        )
        
        # Start the test
        logger.info("🎯 Starting A/B test...")
        success = await ab_tester.start_ab_test(test_config)
        
        if not success:
            logger.error("❌ Failed to start A/B test")
            return
        
        # Initialize mock components
        model = MockModel()
        market_data_gen = MockMarketData()
        
        # Run simulation
        logger.info("🔄 Running simulation...")
        
        # Simulate 50 prediction cycles
        for cycle in range(50):
            logger.info(f"📈 Cycle {cycle + 1}/50")
            
            for exchange in test_config.exchanges:
                # Generate mock prediction
                prediction = model.predict([])
                confidence = model.get_confidence()
                
                # Get market data
                market_data = market_data_gen.get_market_data(exchange)
                
                # Process prediction
                result = await ab_tester.process_prediction(
                    exchange=exchange,
                    prediction=prediction,
                    confidence=confidence,
                    market_data=market_data
                )
                
                # Log result
                logger.info(
                    f"  {exchange}: pred={prediction:.4f}, "
                    f"conf={confidence:.3f}, exec={result.executed}"
                )
            
            # Small delay between cycles
            await asyncio.sleep(0.1)
        
        # Stop the test
        logger.info("🛑 Stopping A/B test...")
        await ab_tester.stop_ab_test()
        
        # Get final status
        status = ab_tester.get_test_status()
        logger.info("📊 Final Test Status:")
        logger.info(f"  Is Running: {status['is_running']}")
        logger.info(f"  Total Results:")
        for exchange, count in status['total_results'].items():
            logger.info(f"    {exchange}: {count} predictions")
        
        # Show performance metrics
        logger.info("🏆 Performance Summary:")
        for exchange, metrics in status['performance_metrics'].items():
            logger.info(
                f"  {exchange}: "
                f"P&L={metrics['total_profit_loss']:.4f}, "
                f"Accuracy={metrics['accuracy']:.3f}, "
                f"ExecRate={metrics['total_executions']/max(metrics['total_predictions'], 1):.3f}"
            )
        
        logger.info("✅ A/B Testing Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in A/B testing demo: {e}")
        import traceback
        traceback.print_exc()


async def run_volume_adaptation_demo():
    """Run a demonstration of volume adaptation."""
    
    logger = system_logger.getChild("VolumeAdaptationDemo")
    logger.info("🌊 Starting Volume Adaptation Demo")
    
    try:
        # Import volume adapter
        from src.supervisor.exchange_volume_adapter import setup_exchange_volume_adapter
        
        # Initialize volume adapter
        config = {
            "exchange_volume_adapter": {
                "enable_volume_adaptation": True,
                "enable_dynamic_adjustment": True
            }
        }
        
        volume_adapter = await setup_exchange_volume_adapter(config)
        
        if not volume_adapter:
            logger.error("❌ Failed to initialize volume adapter")
            return
        
        # Test different scenarios
        test_cases = [
            {"exchange": "BINANCE", "base_size": 0.05, "volume": 1000000, "confidence": 0.8},
            {"exchange": "MEXC", "base_size": 0.05, "volume": 50000, "confidence": 0.8},
            {"exchange": "GATEIO", "base_size": 0.05, "volume": 30000, "confidence": 0.8},
            {"exchange": "BINANCE", "base_size": 0.05, "volume": 1000000, "confidence": 0.5},
            {"exchange": "MEXC", "base_size": 0.05, "volume": 50000, "confidence": 0.5},
        ]
        
        logger.info("📊 Volume Adaptation Results:")
        
        for case in test_cases:
            adjusted_size = volume_adapter.calculate_position_size_adjustment(
                exchange=case["exchange"],
                base_position_size=case["base_size"],
                current_volume=case["volume"],
                confidence_score=case["confidence"]
            )
            
            reduction = (case["base_size"] - adjusted_size) / case["base_size"] * 100
            
            logger.info(
                f"  {case['exchange']}: "
                f"Base={case['base_size']:.3f}, "
                f"Adjusted={adjusted_size:.3f}, "
                f"Reduction={reduction:.1f}%"
            )
        
        logger.info("✅ Volume Adaptation Demo completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in volume adaptation demo: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main demo function."""
    print("🎯 Exchange A/B Testing Framework Demo")
    print("=" * 50)
    
    # Run A/B testing demo
    print("\n1️⃣ Running A/B Testing Demo...")
    await run_ab_test_demo()
    
    print("\n" + "=" * 50)
    
    # Run volume adaptation demo
    print("\n2️⃣ Running Volume Adaptation Demo...")
    await run_volume_adaptation_demo()
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed! Check the logs for detailed results.")


if __name__ == "__main__":
    asyncio.run(main()) 