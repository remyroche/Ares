# src/tactician/dynamic_barrier_calculator.py

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from src.utils.centralized_decorators import (
    handle_errors,
    with_tracing_span,
)
from src.utils.logger import get_logger


class DynamicBarrierCalculator:
    """Dynamic barrier calculator for Tactician based on Analyst triple barrier values.
    
    This calculator dynamically loads Analyst triple barrier configuration and
    calculates Tactician barriers as fractions of those values. It supports
    both 1m and 5m timeframes with appropriate adjustments.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the dynamic barrier calculator."""
        self.config = config.get("tactician_triple_barrier", {})
        self.logger = get_logger("DynamicBarrierCalculator")
        
        # Load Analyst configuration
        self.analyst_config = self._load_analyst_config()
        
        # Load Tactician configuration
        self.tactician_config = self.config
        
        # Initialize dynamic barriers
        self._initialize_dynamic_barriers()

    def _load_analyst_config(self) -> Dict[str, Any]:
        """Load Analyst triple barrier configuration."""
        try:
            # Try to load from Analyst triple barrier labeling component
            analyst_config_path = Path("src/training/steps/step4_analyst_labeling_feature_engineering_components/optimized_triple_barrier_labeling.py")
            
            # Default Analyst values (from the optimized_triple_barrier_labeling.py)
            analyst_config = {
                "profit_take_multiplier": 0.002,  # 0.2%
                "stop_loss_multiplier": 0.001,    # 0.1%
                "time_barrier_minutes": 30,       # 30 minutes
                "max_lookahead": 100,             # 100 periods
                "binary_classification": True
            }
            
            # Try to load from configuration files
            config_paths = [
                "src/config/config.yaml",
                "src/config/enhanced_prediction_integration.yaml",
                "src/config/tactician_triple_barrier_config.yaml"
            ]
            
            for config_path in config_paths:
                if Path(config_path).exists():
                    try:
                        with open(config_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                            
                        # Extract Analyst triple barrier settings if available
                        if "analyst" in config_data:
                            analyst_section = config_data["analyst"]
                            if "triple_barrier" in analyst_section:
                                analyst_config.update(analyst_section["triple_barrier"])
                            elif "profit_take_multiplier" in analyst_section:
                                analyst_config["profit_take_multiplier"] = analyst_section["profit_take_multiplier"]
                            elif "stop_loss_multiplier" in analyst_section:
                                analyst_config["stop_loss_multiplier"] = analyst_section["stop_loss_multiplier"]
                                
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not load Analyst config from {config_path}: {e}")
                        continue
            
            self.logger.info(f"📊 Loaded Analyst Configuration:")
            self.logger.info(f"   Profit Take: {analyst_config['profit_take_multiplier']:.4f} ({analyst_config['profit_take_multiplier']*100:.3f}%)")
            self.logger.info(f"   Stop Loss: {analyst_config['stop_loss_multiplier']:.4f} ({analyst_config['stop_loss_multiplier']*100:.3f}%)")
            self.logger.info(f"   Time Barrier: {analyst_config['time_barrier_minutes']} minutes")
            
            return analyst_config
            
        except Exception as e:
            self.logger.error(f"❌ Error loading Analyst configuration: {e}")
            # Fallback to default values
            return {
                "profit_take_multiplier": 0.002,
                "stop_loss_multiplier": 0.001,
                "time_barrier_minutes": 30,
                "max_lookahead": 100,
                "binary_classification": True
            }

    def _initialize_dynamic_barriers(self) -> None:
        """Initialize dynamic barrier calculation parameters."""
        # Get fractions from configuration
        fractions = self.tactician_config.get("analyst_barrier_fractions", {})
        self.profit_take_fraction = fractions.get("profit_take_fraction", 0.5)
        self.stop_loss_fraction = fractions.get("stop_loss_fraction", 0.25)
        self.time_barrier_fraction = fractions.get("time_barrier_fraction", 0.5)
        
        # Get timeframe settings - both timeframes are equal
        self.timeframes = self.tactician_config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = self.tactician_config.get("primary_timeframe", "1m")
        self.secondary_timeframe = self.tactician_config.get("secondary_timeframe", "5m")
        
        self.logger.info(f"🔧 Dynamic Barrier Calculator Initialized:")
        self.logger.info(f"   Profit Take Fraction: {self.profit_take_fraction:.2f}")
        self.logger.info(f"   Stop Loss Fraction: {self.stop_loss_fraction:.2f}")
        self.logger.info(f"   Time Barrier Fraction: {self.time_barrier_fraction:.2f}")
        self.logger.info(f"   Timeframes: {self.timeframes} (both equal, ML model decides usage)")
        self.logger.info(f"   Primary: {self.primary_timeframe}, Secondary: {self.secondary_timeframe}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=(0.001, 0.00025, 15),
        context="dynamic_barrier_calculator.calculate_dynamic_barriers"
    )
    @with_tracing_span("DynamicBarrier.calculateBarriers")
    def calculate_dynamic_barriers(
        self, 
        timeframe: str = "1m"
    ) -> Tuple[float, float, int]:
        """Calculate dynamic barriers for Tactician based on Analyst values and timeframe.
        
        Args:
            timeframe: The timeframe for calculation ("1m" or "5m")
            
        Returns:
            Tuple of (profit_take_pct, stop_loss_pct, time_barrier_periods)
        """
        try:
            # Validate timeframe
            if timeframe not in self.timeframes:
                self.logger.warning(f"⚠️ Invalid timeframe {timeframe}, using primary timeframe {self.primary_timeframe}")
                timeframe = self.primary_timeframe
            
            # Get Analyst base values
            analyst_pt = self.analyst_config["profit_take_multiplier"]
            analyst_sl = self.analyst_config["stop_loss_multiplier"]
            analyst_time = self.analyst_config["time_barrier_minutes"]
            
            # Calculate base Tactician barriers as fractions of Analyst barriers
            base_tactician_pt = analyst_pt * self.profit_take_fraction
            base_tactician_sl = analyst_sl * self.stop_loss_fraction
            base_tactician_time = int(analyst_time * self.time_barrier_fraction)
            
            # No timeframe-specific adjustments - both timeframes use same fractions
            tactician_pt = base_tactician_pt
            tactician_sl = base_tactician_sl
            
            # Time barrier is adjusted differently (periods vs minutes)
            if timeframe == "1m":
                tactician_time = base_tactician_time  # Same as minutes for 1m
            elif timeframe == "5m":
                tactician_time = int(base_tactician_time / 5)  # Convert to 5m periods
            else:
                tactician_time = base_tactician_time
            
            # No real-time adaptation - barriers are only fractions of Analyst barriers
            # Let the ML model handle market condition adaptation
            
            self.logger.info(f"🎯 Dynamic Barriers Calculated for {timeframe}:")
            self.logger.info(f"   Analyst Base - PT: {analyst_pt:.4f}, SL: {analyst_sl:.4f}, Time: {analyst_time}m")
            self.logger.info(f"   Tactician - PT: {tactician_pt:.4f}, SL: {tactician_sl:.4f}, Time: {tactician_time} periods")
            self.logger.info(f"   Fractions - PT: {self.profit_take_fraction:.2f}, SL: {self.stop_loss_fraction:.2f}, Time: {self.time_barrier_fraction:.2f}")
            self.logger.info(f"   Note: No real-time adaptation - only fractions of Analyst barriers")
            
            return tactician_pt, tactician_sl, tactician_time
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating dynamic barriers: {e}")
            # Return fallback values
            return 0.001, 0.00025, 15

    # Removed volatility and market condition adjustment methods
    # Barriers are only fractions of Analyst barriers - no real-time adaptation

    def get_timeframe_weights(self, timeframe: str) -> Tuple[float, float]:
        """Get execution and confirmation weights for a timeframe."""
        # Both timeframes are equal - let the ML model decide usage
        return 0.5, 0.5

    def calculate_multi_timeframe_barriers(self) -> Dict[str, Tuple[float, float, int]]:
        """Calculate barriers for both 1m and 5m timeframes."""
        try:
            barriers = {}
            
            # Calculate 1m barriers
            if "1m" in self.timeframes:
                barriers["1m"] = self.calculate_dynamic_barriers(timeframe="1m")
            
            # Calculate 5m barriers
            if "5m" in self.timeframes:
                barriers["5m"] = self.calculate_dynamic_barriers(timeframe="5m")
            
            self.logger.info(f"📊 Multi-timeframe barriers calculated:")
            for tf, (pt, sl, time) in barriers.items():
                self.logger.info(f"   {tf}: PT={pt:.4f}, SL={sl:.4f}, Time={time} periods")
            
            return barriers
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating multi-timeframe barriers: {e}")
            return {}

    def get_analyst_barrier_info(self) -> Dict[str, Any]:
        """Get information about Analyst barriers for comparison."""
        return {
            "profit_take_multiplier": self.analyst_config["profit_take_multiplier"],
            "stop_loss_multiplier": self.analyst_config["stop_loss_multiplier"],
            "time_barrier_minutes": self.analyst_config["time_barrier_minutes"],
            "fractions": {
                "profit_take_fraction": self.profit_take_fraction,
                "stop_loss_fraction": self.stop_loss_fraction,
                "time_barrier_fraction": self.time_barrier_fraction
            }
        }

    def validate_barrier_calculation(self, timeframe: str) -> Dict[str, Any]:
        """Validate barrier calculation for a timeframe."""
        try:
            # Calculate barriers
            pt, sl, time = self.calculate_dynamic_barriers(timeframe)
            
            # Get Analyst values
            analyst_pt = self.analyst_config["profit_take_multiplier"]
            analyst_sl = self.analyst_config["stop_loss_multiplier"]
            analyst_time = self.analyst_config["time_barrier_minutes"]
            
            # Calculate actual fractions
            actual_pt_fraction = pt / analyst_pt
            actual_sl_fraction = sl / analyst_sl
            actual_time_fraction = time / analyst_time
            
            # Validate against expected fractions
            pt_fraction_error = abs(actual_pt_fraction - self.profit_take_fraction)
            sl_fraction_error = abs(actual_sl_fraction - self.stop_loss_fraction)
            time_fraction_error = abs(actual_time_fraction - self.time_barrier_fraction)
            
            validation_result = {
                "timeframe": timeframe,
                "analyst_values": {
                    "profit_take": analyst_pt,
                    "stop_loss": analyst_sl,
                    "time_barrier": analyst_time
                },
                "tactician_values": {
                    "profit_take": pt,
                    "stop_loss": sl,
                    "time_barrier": time
                },
                "actual_fractions": {
                    "profit_take": actual_pt_fraction,
                    "stop_loss": actual_sl_fraction,
                    "time_barrier": actual_time_fraction
                },
                "expected_fractions": {
                    "profit_take": self.profit_take_fraction,
                    "stop_loss": self.stop_loss_fraction,
                    "time_barrier": self.time_barrier_fraction
                },
                "fraction_errors": {
                    "profit_take": pt_fraction_error,
                    "stop_loss": sl_fraction_error,
                    "time_barrier": time_fraction_error
                },
                "is_valid": pt_fraction_error < 0.1 and sl_fraction_error < 0.1 and time_fraction_error < 0.1
            }
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating barrier calculation: {e}")
            return {
                "timeframe": timeframe,
                "error": str(e),
                "is_valid": False
            }