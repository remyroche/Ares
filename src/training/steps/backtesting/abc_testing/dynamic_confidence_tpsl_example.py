"""
Dynamic Confidence TPSL Example

This example demonstrates how to use the enhanced confidence-based TPSL system
with real-time confidence score updates from analysts and tacticians.

Key Features Demonstrated:
- Real-time confidence score updates
- Dynamic TPSL adjustment based on confidence changes
- Position tracking with confidence history
- TPSL update history and performance metrics
- Callback system for confidence updates
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import time
import random

# Import enhanced framework components
from .enhanced_abc_testing_framework import (
    TPSLManager, TPSLConfig, TPSLStrategy, ActivePosition, TPSLResult
)
from .paper_trading_engine import MarketData, OrderSide

logger = logging.getLogger(__name__)


class DynamicConfidenceTPSLExample:
    """Example demonstrating dynamic confidence-based TPSL updates."""
    
    def __init__(self):
        """Initialize the dynamic confidence TPSL example."""
        self.logger = logger.getChild('DynamicConfidenceTPSLExample')
        
        # Initialize TPSL manager with confidence-based strategy
        self.tpsl_config = TPSLConfig(
            strategy=TPSLStrategy.CONFIDENCE_BASED,
            take_profit_pct=0.02,                    # Base 2% take profit
            stop_loss_pct=0.01,                      # Base 1% stop loss
            confidence_threshold_high=0.8,           # High confidence threshold
            confidence_threshold_medium=0.6,         # Medium confidence threshold
            confidence_threshold_low=0.4,            # Low confidence threshold
            high_confidence_tp_multiplier=1.5,       # 1.5x TP for high confidence
            high_confidence_sl_multiplier=0.8,       # 0.8x SL for high confidence
            medium_confidence_tp_multiplier=1.0,     # 1.0x TP for medium confidence
            medium_confidence_sl_multiplier=1.0,     # 1.0x SL for medium confidence
            low_confidence_tp_multiplier=0.8,        # 0.8x TP for low confidence
            low_confidence_sl_multiplier=1.2,        # 1.2x SL for low confidence
            analyst_confidence_weight=0.6,           # 60% weight for analyst confidence
            tactician_confidence_weight=0.4,         # 40% weight for tactician confidence
            enable_dynamic_confidence_updates=True,  # Enable real-time updates
            min_confidence_change_threshold=0.05     # 5% minimum change to trigger update
        )
        
        self.tpsl_manager = TPSLManager(self.tpsl_config)
        
        # Register callback for confidence updates
        self.tpsl_manager.register_confidence_update_callback(self._on_confidence_update)
        
        # Simulation data
        self.symbol = "BTCUSDT"
        self.base_price = 50000.0
        self.current_price = self.base_price
        
        self.logger.info("🚀 Dynamic Confidence TPSL Example initialized")
        self.logger.info(f"📊 Strategy: {self.tpsl_config.strategy.value}")
        self.logger.info(f"📊 Dynamic updates: {self.tpsl_config.enable_dynamic_confidence_updates}")
    
    def _on_confidence_update(self, symbol: str, analyst_confidence: float, 
                            tactician_confidence: float, updated_positions: List[str]) -> None:
        """Callback function for confidence updates."""
        self.logger.info(f"🔔 Confidence update callback triggered for {symbol}")
        self.logger.info(f"📊 Analyst: {analyst_confidence:.2f}, Tactician: {tactician_confidence:.2f}")
        self.logger.info(f"📊 Updated positions: {len(updated_positions)}")
        
        # You could add custom logic here, such as:
        # - Sending notifications
        # - Updating external systems
        # - Logging to databases
        # - Triggering additional actions
    
    def _create_market_data(self, price: float, analyst_confidence: float, 
                          tactician_confidence: float) -> MarketData:
        """Create market data with confidence scores."""
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.now(),
            bid_price=price * 0.9999,
            ask_price=price * 1.0001,
            bid_size=1000,
            ask_size=1000,
            last_price=price,
            volume=1000000,
            volatility=0.02,
            spread=0.0002,
            market_condition="normal",
            analyst_confidence=analyst_confidence,
            tactician_confidence=tactician_confidence
        )
    
    def _simulate_price_movement(self, base_price: float, volatility: float = 0.01) -> float:
        """Simulate price movement."""
        # Simple random walk
        change = random.gauss(0, volatility)
        new_price = base_price * (1 + change)
        return max(new_price, base_price * 0.95)  # Prevent extreme drops
    
    def _simulate_confidence_evolution(self, initial_analyst: float, initial_tactician: float,
                                     trend: str = "improving") -> Tuple[float, float]:
        """Simulate confidence score evolution."""
        if trend == "improving":
            analyst_change = random.uniform(0.01, 0.05)
            tactician_change = random.uniform(0.01, 0.03)
        elif trend == "declining":
            analyst_change = random.uniform(-0.05, -0.01)
            tactician_change = random.uniform(-0.03, -0.01)
        else:  # "stable"
            analyst_change = random.uniform(-0.02, 0.02)
            tactician_change = random.uniform(-0.02, 0.02)
        
        new_analyst = max(0.0, min(1.0, initial_analyst + analyst_change))
        new_tactician = max(0.0, min(1.0, initial_tactician + tactician_change))
        
        return new_analyst, new_tactician
    
    async def run_dynamic_confidence_simulation(self, duration_hours: float = 2.0) -> Dict[str, Any]:
        """Run a simulation of dynamic confidence-based TPSL."""
        try:
            self.logger.info(f"🚀 Starting dynamic confidence simulation for {duration_hours} hours")
            
            # Initial confidence scores
            analyst_confidence = 0.7
            tactician_confidence = 0.6
            
            # Create initial position
            market_data = self._create_market_data(self.current_price, analyst_confidence, tactician_confidence)
            position_id = self.tpsl_manager.create_position(
                symbol=self.symbol,
                entry_price=self.current_price,
                position_side=OrderSide.BUY,
                quantity=1.0,
                market_data=market_data
            )
            
            if not position_id:
                raise Exception("Failed to create position")
            
            self.logger.info(f"✅ Position created: {position_id[:8]}...")
            
            # Simulation parameters
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)
            update_interval = timedelta(minutes=15)  # Update every 15 minutes
            next_update = start_time + update_interval
            
            # Track simulation data
            simulation_data = {
                "timestamps": [],
                "prices": [],
                "analyst_confidence": [],
                "tactician_confidence": [],
                "tp_prices": [],
                "sl_prices": [],
                "confidence_levels": [],
                "tpsl_updates": []
            }
            
            # Initial data point
            position = self.tpsl_manager.get_active_positions()[position_id]
            simulation_data["timestamps"].append(start_time)
            simulation_data["prices"].append(self.current_price)
            simulation_data["analyst_confidence"].append(analyst_confidence)
            simulation_data["tactician_confidence"].append(tactician_confidence)
            simulation_data["tp_prices"].append(position.current_tp_price)
            simulation_data["sl_prices"].append(position.current_sl_price)
            simulation_data["confidence_levels"].append(self._get_confidence_level(analyst_confidence, tactician_confidence))
            simulation_data["tpsl_updates"].append(0)
            
            # Simulation loop
            current_time = start_time
            confidence_trends = ["improving", "declining", "stable"]
            current_trend = "stable"
            
            while current_time < end_time:
                # Simulate price movement
                self.current_price = self._simulate_price_movement(self.current_price)
                
                # Check if it's time for confidence update
                if current_time >= next_update:
                    # Change confidence trend occasionally
                    if random.random() < 0.3:  # 30% chance to change trend
                        current_trend = random.choice(confidence_trends)
                    
                    # Simulate confidence evolution
                    analyst_confidence, tactician_confidence = self._simulate_confidence_evolution(
                        analyst_confidence, tactician_confidence, current_trend
                    )
                    
                    # Update confidence scores
                    market_data = self._create_market_data(self.current_price, analyst_confidence, tactician_confidence)
                    self.tpsl_manager.update_confidence_scores(
                        symbol=self.symbol,
                        analyst_confidence=analyst_confidence,
                        tactician_confidence=tactician_confidence,
                        market_data=market_data
                    )
                    
                    # Get updated position
                    position = self.tpsl_manager.get_active_positions()[position_id]
                    
                    # Record data
                    simulation_data["timestamps"].append(current_time)
                    simulation_data["prices"].append(self.current_price)
                    simulation_data["analyst_confidence"].append(analyst_confidence)
                    simulation_data["tactician_confidence"].append(tactician_confidence)
                    simulation_data["tp_prices"].append(position.current_tp_price)
                    simulation_data["sl_prices"].append(position.current_sl_price)
                    simulation_data["confidence_levels"].append(self._get_confidence_level(analyst_confidence, tactician_confidence))
                    simulation_data["tpsl_updates"].append(len(position.tpsl_update_history))
                    
                    # Log update
                    weighted_confidence = (
                        analyst_confidence * self.tpsl_config.analyst_confidence_weight +
                        tactician_confidence * self.tpsl_config.tactician_confidence_weight
                    )
                    confidence_level = self._get_confidence_level(analyst_confidence, tactician_confidence)
                    
                    self.logger.info(f"🔄 Confidence update at {current_time.strftime('%H:%M:%S')}")
                    self.logger.info(f"📊 Trend: {current_trend}")
                    self.logger.info(f"📊 Price: ${self.current_price:.2f}")
                    self.logger.info(f"📊 Analyst: {analyst_confidence:.2f}, Tactician: {tactician_confidence:.2f}")
                    self.logger.info(f"📊 Weighted: {weighted_confidence:.2f} ({confidence_level})")
                    self.logger.info(f"📊 TP: ${position.current_tp_price:.2f}, SL: ${position.current_sl_price:.2f}")
                    self.logger.info(f"📊 TPSL Updates: {len(position.tpsl_update_history)}")
                    
                    next_update = current_time + update_interval
                
                # Advance time
                current_time += timedelta(minutes=5)  # 5-minute time steps
                await asyncio.sleep(0.1)  # Small delay for simulation
            
            # Close position
            exit_price = self.current_price
            exit_reason = "simulation_end"
            result = self.tpsl_manager.close_position(position_id, exit_price, exit_reason)
            
            if result:
                self.logger.info(f"✅ Position closed: {exit_reason}")
                self.logger.info(f"📊 Final P&L: ${result.profit_loss:.2f} ({result.profit_loss_pct:.2%})")
                self.logger.info(f"📊 Total confidence updates: {result.confidence_updates_count}")
                self.logger.info(f"📊 Total TPSL updates: {result.tpsl_updates_count}")
            
            # Get performance metrics
            performance_metrics = self.tpsl_manager.get_tpsl_performance_metrics()
            
            return {
                "simulation_data": simulation_data,
                "final_result": result,
                "performance_metrics": performance_metrics,
                "duration_hours": duration_hours
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in dynamic confidence simulation: {e}")
            raise
    
    def _get_confidence_level(self, analyst_confidence: float, tactician_confidence: float) -> str:
        """Get confidence level based on weighted score."""
        weighted_confidence = (
            analyst_confidence * self.tpsl_config.analyst_confidence_weight +
            tactician_confidence * self.tpsl_config.tactician_confidence_weight
        )
        
        if weighted_confidence >= self.tpsl_config.confidence_threshold_high:
            return "HIGH"
        elif weighted_confidence >= self.tpsl_config.confidence_threshold_medium:
            return "MEDIUM"
        else:
            return "LOW"
    
    def analyze_simulation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze simulation results."""
        try:
            simulation_data = results["simulation_data"]
            performance_metrics = results["performance_metrics"]
            
            # Calculate statistics
            total_updates = len(simulation_data["timestamps"]) - 1  # Exclude initial point
            confidence_changes = len([i for i in range(1, len(simulation_data["confidence_levels"])) 
                                    if simulation_data["confidence_levels"][i] != simulation_data["confidence_levels"][i-1]])
            
            # Price movement analysis
            price_changes = [simulation_data["prices"][i] - simulation_data["prices"][i-1] 
                           for i in range(1, len(simulation_data["prices"]))]
            avg_price_change = np.mean(price_changes)
            price_volatility = np.std(price_changes)
            
            # TPSL adjustment analysis
            tp_changes = [simulation_data["tp_prices"][i] - simulation_data["tp_prices"][i-1] 
                         for i in range(1, len(simulation_data["tp_prices"]))]
            sl_changes = [simulation_data["sl_prices"][i] - simulation_data["sl_prices"][i-1] 
                         for i in range(1, len(simulation_data["sl_prices"]))]
            
            avg_tp_change = np.mean(tp_changes)
            avg_sl_change = np.mean(sl_changes)
            
            # Confidence level distribution
            confidence_levels = simulation_data["confidence_levels"]
            high_count = confidence_levels.count("HIGH")
            medium_count = confidence_levels.count("MEDIUM")
            low_count = confidence_levels.count("LOW")
            
            analysis = {
                "simulation_summary": {
                    "duration_hours": results["duration_hours"],
                    "total_data_points": len(simulation_data["timestamps"]),
                    "confidence_updates": total_updates,
                    "confidence_level_changes": confidence_changes,
                    "final_price": simulation_data["prices"][-1],
                    "price_change_pct": (simulation_data["prices"][-1] - simulation_data["prices"][0]) / simulation_data["prices"][0] * 100
                },
                "confidence_analysis": {
                    "high_confidence_periods": high_count,
                    "medium_confidence_periods": medium_count,
                    "low_confidence_periods": low_count,
                    "avg_analyst_confidence": np.mean(simulation_data["analyst_confidence"]),
                    "avg_tactician_confidence": np.mean(simulation_data["tactician_confidence"]),
                    "confidence_volatility": np.std(simulation_data["analyst_confidence"]) + np.std(simulation_data["tactician_confidence"])
                },
                "tpsl_analysis": {
                    "avg_tp_change": avg_tp_change,
                    "avg_sl_change": avg_sl_change,
                    "tp_volatility": np.std(tp_changes),
                    "sl_volatility": np.std(sl_changes),
                    "max_tp": max(simulation_data["tp_prices"]),
                    "min_tp": min(simulation_data["tp_prices"]),
                    "max_sl": max(simulation_data["sl_prices"]),
                    "min_sl": min(simulation_data["sl_prices"])
                },
                "performance_metrics": performance_metrics
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing simulation results: {e}")
            return {}


async def run_dynamic_confidence_example():
    """Run the dynamic confidence TPSL example."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Dynamic Confidence TPSL Example")
    
    try:
        # Initialize example
        example = DynamicConfidenceTPSLExample()
        
        # Run simulation
        results = await example.run_dynamic_confidence_simulation(duration_hours=1.0)
        
        # Analyze results
        analysis = example.analyze_simulation_results(results)
        
        # Print summary
        logger.info("📊 Dynamic Confidence TPSL Simulation Results:")
        logger.info(f"   Duration: {analysis['simulation_summary']['duration_hours']:.1f} hours")
        logger.info(f"   Data Points: {analysis['simulation_summary']['total_data_points']}")
        logger.info(f"   Confidence Updates: {analysis['simulation_summary']['confidence_updates']}")
        logger.info(f"   Confidence Level Changes: {analysis['simulation_summary']['confidence_level_changes']}")
        logger.info(f"   Price Change: {analysis['simulation_summary']['price_change_pct']:.2f}%")
        
        logger.info("📊 Confidence Analysis:")
        logger.info(f"   High Confidence Periods: {analysis['confidence_analysis']['high_confidence_periods']}")
        logger.info(f"   Medium Confidence Periods: {analysis['confidence_analysis']['medium_confidence_periods']}")
        logger.info(f"   Low Confidence Periods: {analysis['confidence_analysis']['low_confidence_periods']}")
        logger.info(f"   Avg Analyst Confidence: {analysis['confidence_analysis']['avg_analyst_confidence']:.2f}")
        logger.info(f"   Avg Tactician Confidence: {analysis['confidence_analysis']['avg_tactician_confidence']:.2f}")
        
        logger.info("📊 TPSL Analysis:")
        logger.info(f"   Avg TP Change: ${analysis['tpsl_analysis']['avg_tp_change']:.2f}")
        logger.info(f"   Avg SL Change: ${analysis['tpsl_analysis']['avg_sl_change']:.2f}")
        logger.info(f"   TP Range: ${analysis['tpsl_analysis']['min_tp']:.2f} - ${analysis['tpsl_analysis']['max_tp']:.2f}")
        logger.info(f"   SL Range: ${analysis['tpsl_analysis']['min_sl']:.2f} - ${analysis['tpsl_analysis']['max_sl']:.2f}")
        
        if results["final_result"]:
            final_result = results["final_result"]
            logger.info("📊 Final Trade Result:")
            logger.info(f"   Entry Price: ${final_result.entry_price:.2f}")
            logger.info(f"   Exit Price: ${final_result.exit_price:.2f}")
            logger.info(f"   P&L: ${final_result.profit_loss:.2f} ({final_result.profit_loss_pct:.2%})")
            logger.info(f"   Hold Time: {final_result.hold_time_hours:.2f} hours")
            logger.info(f"   Confidence Updates: {final_result.confidence_updates_count}")
            logger.info(f"   TPSL Updates: {final_result.tpsl_updates_count}")
        
        logger.info("✅ Dynamic Confidence TPSL Example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in dynamic confidence example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_dynamic_confidence_example())