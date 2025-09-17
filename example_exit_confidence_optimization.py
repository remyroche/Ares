#!/usr/bin/env python3
"""
Example: Exit Confidence Optimization System

This script demonstrates the complete implementation of the position exit logic
based on analyst and tactician confidence thresholds, with backtesting optimization
to find the ideal exit confidence threshold and combination methods.

Key Features:
1. Position exit logic in signal generation
2. Exit confidence calculation using multiplicative and logarithmic combinations
3. Position state management
4. Backtesting optimization for exit parameters
5. Comprehensive evaluation of exit strategies
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.trading.signal_generation.signal_pipeline import SignalGenerationPipeline, PositionState
from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
from src.trading.config.trading_config import TradingConfig

class ExitConfidenceOptimizationDemo:
    """
    Demonstration of the complete exit confidence optimization system.
    """
    
    def __init__(self):
        self.logger = logger.getChild('ExitConfidenceDemo')
        
        # Configuration for optimization
        self.optimization_config = {
            'n_trials': 30,
            'timeout': 180,
            'study_name': 'exit_confidence_optimization_demo',
            'use_nonlinear_optimization': True
        }
        
        # Mock calibration results for demonstration
        self.mock_calibration_results = {
            'analyst_confidence': [0.7, 0.8, 0.6, 0.9, 0.5],
            'tactician_confidence': [0.8, 0.9, 0.7, 0.8, 0.6],
            'historical_returns': [0.02, -0.01, 0.03, -0.02, 0.01],
            'position_durations': [5, 3, 8, 2, 6],  # Days
            'successful_exits': [True, False, True, False, True]
        }
    
    async def run_demo(self):
        """Run the complete exit confidence optimization demonstration."""
        try:
            self.logger.info("🚀 Starting Exit Confidence Optimization Demo")
            
            # Step 1: Demonstrate signal generation with exit logic
            await self._demo_signal_generation_with_exits()
            
            # Step 2: Demonstrate exit confidence calculations
            self._demo_exit_confidence_calculations()
            
            # Step 3: Demonstrate backtesting optimization
            await self._demo_backtesting_optimization()
            
            # Step 4: Show optimal parameters and their effectiveness
            await self._demo_optimal_parameters()
            
            self.logger.info("✅ Exit Confidence Optimization Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _demo_signal_generation_with_exits(self):
        """Demonstrate signal generation with position exit logic."""
        self.logger.info("📊 Step 1: Signal Generation with Exit Logic")
        
        # Create trading config
        config = TradingConfig()
        
        # Initialize signal generation pipeline
        pipeline = SignalGenerationPipeline(config)
        
        # Mock market data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='H')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(2000, 2100, 10),
            'high': np.random.uniform(2050, 2150, 10),
            'low': np.random.uniform(1950, 2050, 10),
            'close': np.random.uniform(2000, 2100, 10),
            'volume': np.random.uniform(1000, 5000, 10)
        })
        
        self.logger.info("🔄 Simulating position entry and exit scenarios...")
        
        # Simulate different confidence scenarios
        scenarios = [
            {"analyst": 0.8, "tactician": 0.9, "description": "High confidence - should enter"},
            {"analyst": 0.7, "tactician": 0.8, "description": "Good confidence - stay in position"},
            {"analyst": 0.5, "tactician": 0.6, "description": "Medium confidence - monitor"},
            {"analyst": 0.3, "tactician": 0.4, "description": "Low confidence - should exit"},
            {"analyst": 0.6, "tactician": 0.7, "description": "Recovery - may re-enter"}
        ]
        
        for i, scenario in enumerate(scenarios):
            # Simulate analyst and tactician outputs with varying confidence
            analyst_confidence = scenario["analyst"]
            tactician_confidence = scenario["tactician"]
            
            # Calculate exit confidence using different methods
            exit_confidence_mult = self._calculate_multiplicative_exit_confidence(
                analyst_confidence, tactician_confidence, 0.6, 0.4
            )
            exit_confidence_log = self._calculate_logarithmic_exit_confidence(
                analyst_confidence, tactician_confidence, 0.6, 0.4
            )
            exit_confidence_avg = analyst_confidence * 0.4 + tactician_confidence * 0.6
            
            # Check exit conditions with default threshold of 0.5
            should_exit_mult = exit_confidence_mult < 0.5
            should_exit_log = exit_confidence_log < 0.5
            should_exit_avg = exit_confidence_avg < 0.5
            
            self.logger.info(f"   Scenario {i+1}: {scenario['description']}")
            self.logger.info(f"      Analyst: {analyst_confidence:.3f}, Tactician: {tactician_confidence:.3f}")
            self.logger.info(f"      Exit confidence - Mult: {exit_confidence_mult:.3f} ({'EXIT' if should_exit_mult else 'HOLD'})")
            self.logger.info(f"      Exit confidence - Log:  {exit_confidence_log:.3f} ({'EXIT' if should_exit_log else 'HOLD'})")
            self.logger.info(f"      Exit confidence - Avg:  {exit_confidence_avg:.3f} ({'EXIT' if should_exit_avg else 'HOLD'})")
            self.logger.info("")
    
    def _demo_exit_confidence_calculations(self):
        """Demonstrate different exit confidence calculation methods."""
        self.logger.info("🧮 Step 2: Exit Confidence Calculation Methods")
        
        test_cases = [
            {"analyst": 0.8, "tactician": 0.9, "name": "High confidence"},
            {"analyst": 0.6, "tactician": 0.7, "name": "Medium confidence"},
            {"analyst": 0.4, "tactician": 0.5, "name": "Low confidence"},
            {"analyst": 0.2, "tactician": 0.3, "name": "Very low confidence"}
        ]
        
        weights = [
            {"tactician": 0.6, "analyst": 0.4, "name": "Tactician-heavy"},
            {"tactician": 0.5, "analyst": 0.5, "name": "Balanced"},
            {"tactician": 0.4, "analyst": 0.6, "name": "Analyst-heavy"}
        ]
        
        for case in test_cases:
            self.logger.info(f"   {case['name']}: Analyst={case['analyst']:.3f}, Tactician={case['tactician']:.3f}")
            
            for weight in weights:
                mult_conf = self._calculate_multiplicative_exit_confidence(
                    case['analyst'], case['tactician'], weight['tactician'], weight['analyst']
                )
                log_conf = self._calculate_logarithmic_exit_confidence(
                    case['analyst'], case['tactician'], weight['tactician'], weight['analyst']
                )
                avg_conf = case['analyst'] * weight['analyst'] + case['tactician'] * weight['tactician']
                
                self.logger.info(f"      {weight['name']}: Mult={mult_conf:.3f}, Log={log_conf:.3f}, Avg={avg_conf:.3f}")
            self.logger.info("")
    
    async def _demo_backtesting_optimization(self):
        """Demonstrate backtesting optimization for exit parameters."""
        self.logger.info("🔬 Step 3: Backtesting Optimization")
        
        # Create optimizer
        optimizer = FinalParametersOptimizer(self.optimization_config)
        
        # Focus on confidence category optimization
        confidence_params = {
            'exit_confidence_threshold': 0.45,
            'tactician_exit_confidence_weight': 0.65,
            'analyst_exit_confidence_weight': 0.35,
            'exit_confidence_combination_method': 'multiplicative'
        }
        
        self.logger.info("🎯 Evaluating exit strategy performance...")
        
        # Evaluate the exit strategy performance
        exit_score = optimizer._evaluate_exit_strategy_performance(
            confidence_params, self.mock_calibration_results
        )
        
        self.logger.info(f"   Exit strategy score: {exit_score:.3f}")
        
        # Test different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        methods = ['multiplicative', 'logarithmic', 'weighted_average']
        
        self.logger.info("   Testing different configurations:")
        
        best_score = 0.0
        best_config = None
        
        for threshold in thresholds:
            for method in methods:
                test_params = confidence_params.copy()
                test_params['exit_confidence_threshold'] = threshold
                test_params['exit_confidence_combination_method'] = method
                
                score = optimizer._evaluate_exit_strategy_performance(
                    test_params, self.mock_calibration_results
                )
                
                self.logger.info(f"      Threshold={threshold:.1f}, Method={method}: Score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_config = test_params.copy()
        
        self.logger.info(f"   🏆 Best configuration (Score: {best_score:.3f}):")
        if best_config:
            for key, value in best_config.items():
                self.logger.info(f"      {key}: {value}")
        self.logger.info("")
    
    async def _demo_optimal_parameters(self):
        """Demonstrate the effectiveness of optimal parameters."""
        self.logger.info("🎖️ Step 4: Optimal Parameters Demonstration")
        
        # Optimal parameters found through optimization
        optimal_params = {
            'exit_confidence_threshold': 0.45,
            'tactician_exit_confidence_weight': 0.65,
            'analyst_exit_confidence_weight': 0.35,
            'exit_confidence_combination_method': 'multiplicative'
        }
        
        # Default parameters for comparison
        default_params = {
            'exit_confidence_threshold': 0.5,
            'tactician_exit_confidence_weight': 0.6,
            'analyst_exit_confidence_weight': 0.4,
            'exit_confidence_combination_method': 'weighted_average'
        }
        
        # Test scenarios
        test_scenarios = [
            {"analyst_seq": [0.8, 0.7, 0.6, 0.4, 0.3], "tactician_seq": [0.9, 0.8, 0.6, 0.4, 0.2], "name": "Declining confidence"},
            {"analyst_seq": [0.8, 0.8, 0.7, 0.8, 0.8], "tactician_seq": [0.9, 0.8, 0.8, 0.9, 0.8], "name": "Stable confidence"},
            {"analyst_seq": [0.7, 0.5, 0.8, 0.3, 0.6], "tactician_seq": [0.8, 0.6, 0.9, 0.2, 0.7], "name": "Volatile confidence"}
        ]
        
        for scenario in test_scenarios:
            self.logger.info(f"   Testing scenario: {scenario['name']}")
            
            # Test with optimal parameters
            optimal_exit_point = self._find_exit_point(
                scenario['analyst_seq'], scenario['tactician_seq'], optimal_params
            )
            
            # Test with default parameters
            default_exit_point = self._find_exit_point(
                scenario['analyst_seq'], scenario['tactician_seq'], default_params
            )
            
            self.logger.info(f"      Optimal params exit at step: {optimal_exit_point if optimal_exit_point is not None else 'No exit'}")
            self.logger.info(f"      Default params exit at step: {default_exit_point if default_exit_point is not None else 'No exit'}")
            self.logger.info("")
        
        self.logger.info("💡 Key insights:")
        self.logger.info("   • Multiplicative combination is more sensitive to low confidence")
        self.logger.info("   • Lower exit thresholds prevent premature exits")
        self.logger.info("   • Tactician-heavy weighting responds faster to market changes")
        self.logger.info("   • Optimization finds the sweet spot between sensitivity and stability")
    
    def _find_exit_point(self, analyst_seq: List[float], tactician_seq: List[float], 
                        params: Dict[str, Any]) -> Optional[int]:
        """Find the exit point for a given confidence sequence and parameters."""
        threshold = params['exit_confidence_threshold']
        tactician_weight = params['tactician_exit_confidence_weight']
        analyst_weight = params['analyst_exit_confidence_weight']
        method = params['exit_confidence_combination_method']
        
        for i, (analyst_conf, tactician_conf) in enumerate(zip(analyst_seq, tactician_seq)):
            if method == 'multiplicative':
                exit_conf = self._calculate_multiplicative_exit_confidence(
                    analyst_conf, tactician_conf, tactician_weight, analyst_weight
                )
            elif method == 'logarithmic':
                exit_conf = self._calculate_logarithmic_exit_confidence(
                    analyst_conf, tactician_conf, tactician_weight, analyst_weight
                )
            else:  # weighted_average
                exit_conf = analyst_conf * analyst_weight + tactician_conf * tactician_weight
            
            if exit_conf < threshold:
                return i
        
        return None
    
    def _calculate_multiplicative_exit_confidence(self, analyst_conf: float, tactician_conf: float,
                                                tactician_weight: float, analyst_weight: float) -> float:
        """Calculate exit confidence using multiplicative method."""
        analyst_conf = max(0.001, analyst_conf)
        tactician_conf = max(0.001, tactician_conf)
        
        multiplicative_conf = (
            (tactician_conf ** tactician_weight) * 
            (analyst_conf ** analyst_weight)
        )
        
        return min(1.0, multiplicative_conf)
    
    def _calculate_logarithmic_exit_confidence(self, analyst_conf: float, tactician_conf: float,
                                             tactician_weight: float, analyst_weight: float) -> float:
        """Calculate exit confidence using logarithmic method."""
        analyst_conf = max(0.001, analyst_conf)
        tactician_conf = max(0.001, tactician_conf)
        
        log_combination = (
            tactician_weight * np.log(tactician_conf) +
            analyst_weight * np.log(analyst_conf)
        )
        
        logarithmic_conf = np.exp(log_combination)
        return min(1.0, max(0.0, logarithmic_conf))

async def main():
    """Main function to run the demonstration."""
    demo = ExitConfidenceOptimizationDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())