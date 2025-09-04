# src/tactician/dynamic_barrier_calculator.py

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from src.core.decorators import handles_errors, traced
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
            analyst_config_path = Path("src/training/steps/step06_labeling_components/optimized_triple_barrier_labeling.py")
            
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
        # Get fractions from configuration - 4 barrier combinations
        fractions = self.tactician_config.get("analyst_barrier_fractions", {})
        self.upper_barrier_50_fraction = fractions.get("upper_barrier_50_fraction", 0.5)
        self.lower_barrier_50_fraction = fractions.get("lower_barrier_50_fraction", 0.5)
        self.upper_barrier_25_fraction = fractions.get("upper_barrier_25_fraction", 0.25)
        self.lower_barrier_25_fraction = fractions.get("lower_barrier_25_fraction", 0.25)
        
        # Get timeframe settings - both timeframes are equal
        self.timeframes = self.tactician_config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = self.tactician_config.get("primary_timeframe", "1m")
        self.secondary_timeframe = self.tactician_config.get("secondary_timeframe", "5m")
        
        self.logger.info(f"🔧 Dynamic Barrier Calculator Initialized:")
        self.logger.info(f"   50% Barriers - Upper: {self.upper_barrier_50_fraction:.2f}, Lower: {self.lower_barrier_50_fraction:.2f}")
        self.logger.info(f"   25% Barriers - Upper: {self.upper_barrier_25_fraction:.2f}, Lower: {self.lower_barrier_25_fraction:.2f}")
        self.logger.info(f"   Timeframes: {self.timeframes} (both equal, ML model decides usage)")
        self.logger.info(f"   Primary: {self.primary_timeframe}, Secondary: {self.secondary_timeframe}")

    @handles_errors(default_return=(0.001, 0.00025), context="dynamic_barrier_calculator.calculate_dynamic_barriers" )
    @traced("DynamicBarrier.calculateBarriers")
    def calculate_dynamic_barriers(
        self, 
        timeframe: str = "1m"
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate 2 dynamic barrier combinations for Tactician.
        
        Args:
            timeframe: The timeframe for calculation ("1m" or "5m")
            
        Returns:
            Dictionary with 2 barrier combinations:
            - barrier_50_50: (50% upper, 50% lower)
            - barrier_25_25: (25% upper, 25% lower)
        """
        try:
            # Validate timeframe
            if timeframe not in self.timeframes:
                self.logger.warning(f"⚠️ Invalid timeframe {timeframe}, using primary timeframe {self.primary_timeframe}")
                timeframe = self.primary_timeframe
            
            # Get Analyst base values
            analyst_upper = self.analyst_config["profit_take_multiplier"]  # Upper barrier (profit take)
            analyst_lower = self.analyst_config["stop_loss_multiplier"]    # Lower barrier (stop loss)
            
            # Calculate 2 barrier combinations
            barriers = {
                "barrier_50_50": (
                    analyst_upper * self.upper_barrier_50_fraction,  # 50% upper
                    analyst_lower * self.lower_barrier_50_fraction   # 50% lower
                ),
                "barrier_25_25": (
                    analyst_upper * self.upper_barrier_25_fraction,  # 25% upper
                    analyst_lower * self.lower_barrier_25_fraction   # 25% lower
                )
            }
            
            self.logger.info(f"🎯 2 Dynamic Barrier Combinations Calculated for {timeframe}:")
            self.logger.info(f"   Analyst Base - Upper: {analyst_upper:.4f}, Lower: {analyst_lower:.4f}")
            for name, (upper, lower) in barriers.items():
                self.logger.info(f"   {name}: Upper={upper:.4f}, Lower={lower:.4f}")
            self.logger.info(f"   Note: Both barrier combinations will be used - position only opens if confidence is high enough for both")
            
            return barriers
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating dynamic barriers: {e}")
            # Return fallback values
            return {
                "barrier_50_50": (0.001, 0.0005),
                "barrier_25_25": (0.0005, 0.00025)
            }

    # Removed volatility and market condition adjustment methods
    # Barriers are only fractions of Analyst barriers - no real-time adaptation

    def get_timeframe_weights(self, timeframe: str) -> Tuple[float, float]:
        """Get execution and confirmation weights for a timeframe."""
        # Both timeframes are equal - let the ML model decide usage
        return 0.5, 0.5

    def calculate_multi_timeframe_barriers(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate 2 barrier combinations for both 1m and 5m timeframes."""
        try:
            barriers = {}
            
            # Calculate 1m barriers (2 combinations)
            if "1m" in self.timeframes:
                barriers["1m"] = self.calculate_dynamic_barriers(timeframe="1m")
            
            # Calculate 5m barriers (2 combinations)
            if "5m" in self.timeframes:
                barriers["5m"] = self.calculate_dynamic_barriers(timeframe="5m")
            
            self.logger.info(f"📊 Multi-timeframe barriers calculated (2 combinations each):")
            for tf, combinations in barriers.items():
                self.logger.info(f"   {tf}:")
                for name, (upper, lower) in combinations.items():
                    self.logger.info(f"     {name}: Upper={upper:.4f}, Lower={lower:.4f}")
            self.logger.info(f"   Note: Both barrier combinations will be used for each timeframe")
            
            return barriers
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating multi-timeframe barriers: {e}")
            return {}

    def get_analyst_barrier_info(self) -> Dict[str, Any]:
        """Get information about Analyst barriers for comparison."""
        return {
            "upper_barrier_multiplier": self.analyst_config["profit_take_multiplier"],
            "lower_barrier_multiplier": self.analyst_config["stop_loss_multiplier"],
            "fractions": {
                "upper_barrier_50_fraction": self.upper_barrier_50_fraction,
                "lower_barrier_50_fraction": self.lower_barrier_50_fraction,
                "upper_barrier_25_fraction": self.upper_barrier_25_fraction,
                "lower_barrier_25_fraction": self.lower_barrier_25_fraction
            }
        }

    def validate_barrier_calculation(self, timeframe: str) -> Dict[str, Any]:
        """Validate barrier calculation for a timeframe."""
        try:
            # Calculate 2 barrier combinations
            barriers = self.calculate_dynamic_barriers(timeframe)
            
            # Get Analyst values
            analyst_upper = self.analyst_config["profit_take_multiplier"]
            analyst_lower = self.analyst_config["stop_loss_multiplier"]
            
            validation_results = {}
            
            for barrier_name, (upper, lower) in barriers.items():
                # Calculate actual fractions
                actual_upper_fraction = upper / analyst_upper
                actual_lower_fraction = lower / analyst_lower
                
                # Get expected fractions based on barrier name
                if "50" in barrier_name:
                    expected_upper_fraction = self.upper_barrier_50_fraction
                    expected_lower_fraction = self.lower_barrier_50_fraction
                else:
                    expected_upper_fraction = self.upper_barrier_25_fraction
                    expected_lower_fraction = self.lower_barrier_25_fraction
                
                # Validate against expected fractions
                upper_fraction_error = abs(actual_upper_fraction - expected_upper_fraction)
                lower_fraction_error = abs(actual_lower_fraction - expected_lower_fraction)
                
                validation_results[barrier_name] = {
                    "analyst_values": {
                        "upper_barrier": analyst_upper,
                        "lower_barrier": analyst_lower
                    },
                    "tactician_values": {
                        "upper_barrier": upper,
                        "lower_barrier": lower
                    },
                    "actual_fractions": {
                        "upper_barrier": actual_upper_fraction,
                        "lower_barrier": actual_lower_fraction
                    },
                    "expected_fractions": {
                        "upper_barrier": expected_upper_fraction,
                        "lower_barrier": expected_lower_fraction
                    },
                    "fraction_errors": {
                        "upper_barrier": upper_fraction_error,
                        "lower_barrier": lower_fraction_error
                    },
                    "is_valid": upper_fraction_error < 0.1 and lower_fraction_error < 0.1
                }
            
            return {
                "timeframe": timeframe,
                "barrier_combinations": validation_results,
                "overall_valid": all(result["is_valid"] for result in validation_results.values())
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error validating barrier calculation: {e}")
            return {
                "timeframe": timeframe,
                "error": str(e),
                "is_valid": False
            }