"""S/R Parameter Loader for Live Trading.

This module ensures that optimized S/R parameters from step 2.5 are loaded
and used consistently across both training and live trading.
"""

import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

from src.utils.logger import system_logger


class SRParameterLoader:
    """Loads and manages optimized S/R parameters."""
    
    @staticmethod
    def load_optimized_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load optimized S/R parameters from step 2.5.
        
        Args:
            config: System configuration
            
        Returns:
            Dictionary containing optimized S/R parameters
        """
        logger = system_logger.getChild("SRParameterLoader")
        
        try:
            # Check if parameters are already in config (from training)
            if "sr_probability_calculation" in config and config["sr_probability_calculation"]:
                logger.info("✅ Using S/R parameters from config")
                return config["sr_probability_calculation"]
            
            # Try to load from file
            param_file = Path(config.get("model_save_path", "models")) / "optimized_sr_parameters.json"
            
            if param_file.exists():
                with open(param_file, 'r') as f:
                    data = json.load(f)
                    parameters = data.get("parameters", {})
                    
                    # Update config with loaded parameters
                    config["sr_probability_calculation"] = parameters
                    config.setdefault("sr_breakout_predictor", {})["optimized_parameters"] = parameters
                    config["sr_breakout_predictor"]["use_optimized_params"] = True
                    
                    logger.info(f"✅ Loaded optimized S/R parameters from {param_file}")
                    logger.info(f"   Parameters: {len(parameters)} values loaded")
                    
                    # Log key parameters
                    logger.info(f"   Price action weight: {parameters.get('price_action_weight', 'N/A')}")
                    logger.info(f"   Volume weight: {parameters.get('volume_weight', 'N/A')}")
                    logger.info(f"   Volatility weight: {parameters.get('volatility_weight', 'N/A')}")
                    
                    return parameters
            else:
                logger.warning(f"⚠️ Optimized parameter file not found: {param_file}")
                logger.warning("⚠️ Using default S/R parameters - consider running step 2.5 optimization")
                
                # Return default parameters
                defaults = SRParameterLoader.get_default_parameters()
                config["sr_probability_calculation"] = defaults
                return defaults
                
        except Exception as e:
            logger.error(f"❌ Error loading optimized S/R parameters: {e}")
            logger.warning("⚠️ Falling back to default parameters")
            
            defaults = SRParameterLoader.get_default_parameters()
            config["sr_probability_calculation"] = defaults
            return defaults
    
    @staticmethod
    def get_default_parameters() -> Dict[str, float]:
        """Get default S/R parameters as fallback."""
        return {
            "price_action_weight": 0.3,
            "momentum_weight": 0.2,
            "trend_strength_weight": 0.2,
            "volume_weight": 0.2,
            "volatility_weight": 0.1,
            "volume_surge_multiplier": 2.0,
            "volume_confirmation_threshold": 1.5,
            "high_volatility_breakout_boost": 0.15,
            "low_volatility_consolidation_boost": 0.1,
            "level_strength_weight": 0.2,
            "touch_count_weight": 0.3,
            "age_decay_factor": 0.95,
            "proximity_threshold": 0.002,
            "proximity_decay_rate": 2.0,
            "min_breakout_probability": 0.2,
            "max_breakout_probability": 0.8,
            "default_probability": 0.33
        }
    
    @staticmethod
    def ensure_parameters_loaded(config: Dict[str, Any]) -> None:
        """
        Ensure S/R parameters are loaded in the configuration.
        This should be called during system initialization.
        
        Args:
            config: System configuration dictionary (will be modified in-place)
        """
        logger = system_logger.getChild("SRParameterLoader")
        
        # Check if parameters are already loaded
        if "sr_probability_calculation" in config and config["sr_probability_calculation"]:
            logger.info("✅ S/R parameters already loaded in configuration")
            return
        
        # Load parameters
        logger.info("📂 Loading optimized S/R parameters...")
        SRParameterLoader.load_optimized_parameters(config)
    
    @staticmethod
    def get_parameter_summary(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of loaded S/R parameters.
        
        Args:
            config: System configuration
            
        Returns:
            Summary dictionary with parameter status
        """
        params = config.get("sr_probability_calculation", {})
        
        if not params:
            return {
                "status": "not_loaded",
                "source": "none",
                "parameter_count": 0
            }
        
        # Check if these are optimized or default parameters
        param_file = Path(config.get("model_save_path", "models")) / "optimized_sr_parameters.json"
        source = "optimized" if param_file.exists() else "default"
        
        # Calculate weight sum for validation
        weight_keys = [
            "price_action_weight", "momentum_weight", "trend_strength_weight",
            "volume_weight", "volatility_weight"
        ]
        weight_sum = sum(params.get(k, 0) for k in weight_keys)
        
        return {
            "status": "loaded",
            "source": source,
            "parameter_count": len(params),
            "weight_sum": weight_sum,
            "weights_normalized": abs(weight_sum - 1.0) < 0.01,
            "key_parameters": {
                "price_action_weight": params.get("price_action_weight"),
                "volume_weight": params.get("volume_weight"),
                "proximity_threshold": params.get("proximity_threshold")
            }
        }


def initialize_sr_parameters(config: Dict[str, Any]) -> None:
    """
    Initialize S/R parameters during system startup.
    This is a convenience function that should be called early in initialization.
    
    Args:
        config: System configuration dictionary
    """
    SRParameterLoader.ensure_parameters_loaded(config)