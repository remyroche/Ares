#!/usr/bin/env python3
"""
Configuration Manager for Ares Launcher

This module handles configuration management, environment variables, and training modes,
extracting this functionality from the main launcher class.
"""

import logging
import os
from typing import Dict, Any, Optional

from src.config.training_modes import (
    get_training_mode_config,
    get_intensity_percentage,
    get_intensity_comparison,
    get_mode_recommendations,
    list_available_modes,
)


class EnvironmentManager:
    """Manages environment variables for different training modes."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def set_training_mode(self, training_mode: str) -> None:
        """Set environment variables for the specified training mode."""
        # Clear all training mode flags first
        os.environ.pop("LIGHT_TRAINING_MODE", None)
        os.environ.pop("BLANK_TRAINING_MODE", None)
        os.environ.pop("FULL_TRAINING_MODE", None)
        
        if training_mode == "light":
            os.environ["LIGHT_TRAINING_MODE"] = "1"
            self.logger.info("💡 LIGHT TRAINING MODE: Set LIGHT_TRAINING_MODE=1")
        elif training_mode == "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            self.logger.info("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1")
        elif training_mode == "full":
            os.environ["FULL_TRAINING_MODE"] = "1"
            self.logger.info("📊 FULL TRAINING MODE: Set FULL_TRAINING_MODE=1")
        else:
            self.logger.warning(f"Unknown training mode: {training_mode}")
    
    def set_trading_mode(self, trading_mode: str) -> None:
        """Set environment variable for trading mode."""
        os.environ["TRADING_MODE"] = trading_mode
        self.logger.info(f"📊 TRADING MODE: Set TRADING_MODE={trading_mode}")
    
    def set_force_mode(self, force: bool = True) -> None:
        """Set force mode environment variable."""
        if force:
            os.environ["FORCE"] = "1"
            self.logger.info("🔄 FORCE MODE: Set FORCE=1")
        else:
            os.environ.pop("FORCE", None)
    
    def set_pipeline_mode(self, pipeline_type: str, symbol: str, exchange: str) -> None:
        """Set environment variables for pipeline execution."""
        os.environ[f"{pipeline_type.upper().replace('-', '_')}_MODE"] = "enhanced"
        os.environ["SYMBOL"] = symbol
        os.environ["EXCHANGE"] = exchange
        self.logger.info(f"🔧 Pipeline mode set: {pipeline_type.upper()}_MODE=enhanced")
    
    def clear_pipeline_mode(self, pipeline_type: str) -> None:
        """Clear pipeline mode environment variables."""
        os.environ.pop(f"{pipeline_type.upper().replace('-', '_')}_MODE", None)
        os.environ.pop("SYMBOL", None)
        os.environ.pop("EXCHANGE", None)


class TrainingModeManager:
    """Manages training mode configurations and display."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def get_mode_config(self, training_mode: str) -> Any:
        """Get training mode configuration."""
        try:
            return get_training_mode_config(training_mode)
        except ValueError as e:
            self.logger.error(f"Invalid training mode: {e}")
            raise
    
    def get_mode_display_info(self, training_mode: str) -> Dict[str, Any]:
        """Get display information for a training mode."""
        try:
            config = self.get_mode_config(training_mode)
            intensity_pct = get_intensity_percentage(training_mode) * 100
            
            return {
                "mode": training_mode,
                "intensity_percentage": intensity_pct,
                "lookback_days": config.lookback_days,
                "max_trials": config.max_trials,
                "n_trials": config.n_trials,
                "estimated_duration_minutes": config.estimated_duration_minutes,
                "computational_intensity": config.computational_intensity,
            }
        except ValueError as e:
            self.logger.error(f"Error getting mode display info: {e}")
            return {}
    
    def show_training_modes(self) -> bool:
        """Display available training modes and their configurations."""
        print("=" * 80)
        print("🎯 AVAILABLE TRAINING MODES")
        print("=" * 80)
        
        # Show intensity comparison table
        print("\n📊 INTENSITY COMPARISON")
        print("-" * 80)
        comparison = get_intensity_comparison()
        
        # Print header
        print(f"{'Mode':<8} {'Intensity':<12} {'Max Trials':<12} {'N Trials':<10} {'Duration':<10} {'Lookback':<10}")
        print("-" * 80)
        
        for mode, data in comparison.items():
            intensity_pct = f"{data['intensity_percentage']*100:.0f}%"
            print(f"{mode:<8} {intensity_pct:<12} {data['max_trials']:<12} {data['n_trials']:<10} {data['estimated_duration_minutes']:<10}min {data['lookback_days']:<10}days")
        
        print("\n" + "=" * 80)
        print("📋 DETAILED MODE CONFIGURATIONS")
        print("=" * 80)
        
        modes = list_available_modes()
        recommendations = get_mode_recommendations()
        
        for mode_name, description in modes.items():
            try:
                config = self.get_mode_config(mode_name)
                recommendation = recommendations.get(mode_name, "No specific recommendation available.")
                intensity_pct = f"{get_intensity_percentage(mode_name)*100:.0f}%"
                
                print(f"\n📊 {mode_name.upper()} MODE ({intensity_pct} of full intensity)")
                print(f"   Description: {description}")
                print(f"   Lookback Days: {config.lookback_days}")
                print(f"   Max Trials: {config.max_trials}")
                print(f"   N Trials: {config.n_trials}")
                print(f"   Exclude Recent Days: {config.exclude_recent_days}")
                print(f"   Min Data Points: {config.min_data_points}")
                print(f"   Computational Intensity: {config.computational_intensity}")
                print(f"   Estimated Duration: {config.estimated_duration_minutes} minutes")
                print(f"   Advanced Model Training: {'✅' if config.enable_advanced_model_training else '❌'}")
                print(f"   Ensemble Training: {'✅' if config.enable_ensemble_training else '❌'}")
                print(f"   Multi-timeframe Training: {'✅' if config.enable_multi_timeframe_training else '❌'}")
                print(f"   Adaptive Training: {'✅' if config.enable_adaptive_training else '❌'}")
                print(f"   Recommendation: {recommendation}")
                
            except ValueError as e:
                print(f"\n❌ Error loading {mode_name} mode: {e}")
        
        print("\n" + "=" * 80)
        print("💡 USAGE EXAMPLES")
        print("=" * 80)
        print("  python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE")
        print("  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE")
        print("  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE")
        print("  python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE --lookback-days 15")
        print("  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --lookback-days 90")
        print("  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --lookback-days 365")
        print("=" * 80)
        
        return True


class ConfigurationManager:
    """Main configuration manager that coordinates all configuration aspects."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.environment_manager = EnvironmentManager(logger)
        self.training_mode_manager = TrainingModeManager(logger)
    
    def setup_training_environment(
        self, 
        training_mode: str, 
        symbol: str, 
        exchange: str,
        lookback_days: Optional[int] = None,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Set up environment for training execution."""
        # Set training mode
        self.environment_manager.set_training_mode(training_mode)
        
        # Set force mode if needed
        if force_rerun:
            self.environment_manager.set_force_mode(True)
        
        # Get mode configuration
        try:
            mode_config = self.training_mode_manager.get_mode_config(training_mode)
            
            # Use provided lookback_days or mode default
            actual_lookback_days = lookback_days if lookback_days is not None else mode_config.lookback_days
            
            # Get display info
            display_info = self.training_mode_manager.get_mode_display_info(training_mode)
            
            return {
                "mode_config": mode_config,
                "lookback_days": actual_lookback_days,
                "display_info": display_info,
                "force_rerun": force_rerun,
            }
            
        except ValueError as e:
            self.logger.error(f"Failed to setup training environment: {e}")
            raise
    
    def setup_trading_environment(self, trading_mode: str) -> None:
        """Set up environment for trading execution."""
        self.environment_manager.set_trading_mode(trading_mode)
    
    def setup_pipeline_environment(
        self, 
        pipeline_type: str, 
        symbol: str, 
        exchange: str
    ) -> None:
        """Set up environment for pipeline execution."""
        self.environment_manager.set_pipeline_mode(pipeline_type, symbol, exchange)
    
    def cleanup_pipeline_environment(self, pipeline_type: str) -> None:
        """Clean up pipeline environment variables."""
        self.environment_manager.clear_pipeline_mode(pipeline_type)
    
    def show_training_modes(self) -> bool:
        """Display available training modes."""
        return self.training_mode_manager.show_training_modes()
    
    def get_mode_config(self, training_mode: str) -> Any:
        """Get training mode configuration."""
        return self.training_mode_manager.get_mode_config(training_mode)
    
    def get_mode_display_info(self, training_mode: str) -> Dict[str, Any]:
        """Get display information for a training mode."""
        return self.training_mode_manager.get_mode_display_info(training_mode)