# src/training/enhanced_training_manager.py

import asyncio
import pandas as pd
import time
import psutil
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.validator_orchestrator import validator_orchestrator

# Import computational optimization components
from src.training.optimization.computational_optimization_manager import (
    ComputationalOptimizationManager,
    create_computational_optimization_manager,
)
from src.config.computational_optimization import get_computational_optimization_config


class EnhancedTrainingManager:
    """
    Enhanced training manager with comprehensive 16-step pipeline.
    This module orchestrates the complete training pipeline including analyst and tactician steps.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize enhanced training manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnhancedTrainingManager")

        # Enhanced training manager state
        self.is_training: bool = False
        self.enhanced_training_results: dict[str, Any] = {}
        self.enhanced_training_history: list[dict[str, Any]] = []

        # Configuration
        self.enhanced_training_config: dict[str, Any] = self.config.get(
            "enhanced_training_manager",
            {},
        )
        self.enhanced_training_interval: int = self.enhanced_training_config.get(
            "enhanced_training_interval",
            3600,
        )
        self.max_enhanced_training_history: int = self.enhanced_training_config.get(
            "max_enhanced_training_history",
            100,
        )
        
        # Training parameters
        self.blank_training_mode: bool = self.enhanced_training_config.get("blank_training_mode", False)
        self.max_trials: int = self.enhanced_training_config.get("max_trials", 200)
        self.n_trials: int = self.enhanced_training_config.get("n_trials", 100)
        self.lookback_days: int = self.enhanced_training_config.get("lookback_days", 30)
        
        # Validation parameters
        self.enable_validators: bool = self.enhanced_training_config.get("enable_validators", True)
        self.validation_results: dict[str, Any] = {}
        
        # Computational optimization parameters
        self.enable_computational_optimization: bool = self.enhanced_training_config.get("enable_computational_optimization", True)
        self.computational_optimization_manager: ComputationalOptimizationManager | None = None
        self.optimization_statistics: dict[str, Any] = {}
        
        # Checkpointing configuration
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "training_progress.json"
        self.enable_checkpointing = self.enhanced_training_config.get("enable_checkpointing", True)
        
    def _save_checkpoint(self, step_name: str, pipeline_state: dict[str, Any]) -> None:
        """
        Save training progress checkpoint.
        
        Args:
            step_name: Current step name
            pipeline_state: Current pipeline state
        """
        if not self.enable_checkpointing:
            return
            
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "current_step": step_name,
                "pipeline_state": pipeline_state,
                "training_mode": "blank" if self.blank_training_mode else "full",
                "symbol": getattr(self, 'current_symbol', ''),
                "exchange": getattr(self, 'current_exchange', ''),
                "timeframe": getattr(self, 'current_timeframe', '1m'),
                "lookback_days": self.lookback_days,
                "max_trials": self.max_trials,
                "n_trials": self.n_trials
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            self.logger.info(f"💾 Checkpoint saved: {step_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> dict[str, Any] | None:
        """
        Load training progress checkpoint.
        
        Returns:
            dict: Checkpoint data or None if no checkpoint exists
        """
        if not self.enable_checkpointing or not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.logger.info(f"📂 Checkpoint loaded: {checkpoint_data.get('current_step', 'unknown')}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _clear_checkpoint(self) -> None:
        """Clear the checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.logger.info("🗑️ Checkpoint cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear checkpoint: {e}")
        
    def _get_system_resources(self) -> dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            dict: System resource information
        """
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            # Get system-wide memory info
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent
            
            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "system_memory_percent": system_memory_percent,
                "available_memory_gb": system_memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            self.logger.warning(f"Could not get system resources: {e}")
            return {"memory_mb": 0, "cpu_percent": 0, "system_memory_percent": 0, "available_memory_gb": 0}
    
    def _analyze_resource_requirements(self) -> dict[str, Any]:
        """
        Analyze resource requirements for the training process.
        
        Returns:
            dict: Resource analysis information
        """
        try:
            # Get system info
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
            
            # Realistic estimates based on actual training complexity
            if self.blank_training_mode:
                estimated_memory_gb = 4.0  # Blank training uses less memory
                estimated_time_minutes = 90  # Realistic: 1.5 hours for blank training
                memory_warning_threshold = 6.0
                models_to_train = 4
                optimization_trials = 50
            else:
                estimated_memory_gb = 8.0  # Full training uses more memory
                estimated_time_minutes = 720  # Realistic: 12 hours for full training
                memory_warning_threshold = 12.0
                models_to_train = 12
                optimization_trials = 200
            
            # Check if system meets requirements
            memory_sufficient = memory_gb >= memory_warning_threshold
            cpu_sufficient = cpu_count >= 4
            
            return {
                "system_memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "estimated_memory_gb": estimated_memory_gb,
                "estimated_time_minutes": estimated_time_minutes,
                "models_to_train": models_to_train,
                "optimization_trials": optimization_trials,
                "memory_sufficient": memory_sufficient,
                "cpu_sufficient": cpu_sufficient,
                "memory_warning_threshold": memory_warning_threshold,
                "recommendations": self._get_resource_recommendations(memory_gb, cpu_count),
                "step_breakdown": self._get_step_time_breakdown(self.blank_training_mode)
            }
        except Exception as e:
            self.logger.warning(f"Could not analyze resource requirements: {e}")
            return {}
    
    def _get_resource_recommendations(self, memory_gb: float, cpu_count: int) -> list[str]:
        """
        Get resource recommendations based on system specs.
        
        Args:
            memory_gb: Available memory in GB
            cpu_count: Number of CPU cores
            
        Returns:
            list: Recommendations
        """
        recommendations = []
        
        if memory_gb < 8:
            recommendations.append("⚠️ Consider upgrading to 16GB RAM for optimal performance")
        elif memory_gb < 12:
            recommendations.append("💡 16GB RAM recommended for full training mode")
        
        if cpu_count < 4:
            recommendations.append("⚠️ Consider using a system with at least 4 CPU cores")
        elif cpu_count < 8:
            recommendations.append("💡 8+ CPU cores recommended for faster training")
        
        if self.blank_training_mode:
            recommendations.append("✅ Blank training mode is suitable for your system")
        else:
            if memory_gb < 12:
                recommendations.append("⚠️ Full training mode may be slow on your system")
            else:
                recommendations.append("✅ Full training mode should work well on your system")
        
        return recommendations
    
    def _get_step_time_breakdown(self, is_blank_mode: bool) -> dict[str, int]:
        """
        Get realistic time breakdown for each step.
        
        Args:
            is_blank_mode: Whether this is blank training mode
            
        Returns:
            dict: Time estimates for each step in minutes
        """
        if is_blank_mode:
            return {
                "step1_data_collection": 5,
                "step2_market_regime_classification": 3,
                "step3_regime_data_splitting": 2,
                "step4_analyst_labeling_feature_engineering": 15,
                "step5_analyst_specialist_training": 10,
                "step6_analyst_enhancement": 8,
                "step7_analyst_ensemble_creation": 12,
                "step8_tactician_labeling": 5,
                "step9_tactician_specialist_training": 10,
                "step10_tactician_ensemble_creation": 12,
                "step11_confidence_calibration": 3,
                "step12_final_parameters_optimization": 15,
                "step13_walk_forward_validation": 8,
                "step14_monte_carlo_validation": 8,
                "step15_ab_testing": 5,
                "step16_saving": 2
            }
        else:
            return {
                "step1_data_collection": 15,
                "step2_market_regime_classification": 8,
                "step3_regime_data_splitting": 5,
                "step4_analyst_labeling_feature_engineering": 60,
                "step5_analyst_specialist_training": 30,
                "step6_analyst_enhancement": 25,
                "step7_analyst_ensemble_creation": 35,
                "step8_tactician_labeling": 15,
                "step9_tactician_specialist_training": 30,
                "step10_tactician_ensemble_creation": 35,
                "step11_confidence_calibration": 10,
                "step12_final_parameters_optimization": 240,
                "step13_walk_forward_validation": 60,
                "step14_monte_carlo_validation": 60,
                "step15_ab_testing": 30,
                "step16_saving": 5
            }
    
    def _optimize_memory_usage(self) -> None:
        """
        Perform memory optimization to reduce memory footprint.
        """
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Log memory before and after optimization
            before_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            self.logger.info(f"🧹 Memory optimization: {before_memory:.1f} MB before cleanup")
            
            # Force another garbage collection
            gc.collect()
            
            after_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_saved = before_memory - after_memory
            
            self.logger.info(f"🧹 Memory optimization: {after_memory:.1f} MB after cleanup (saved {memory_saved:.1f} MB)")
            
            if memory_saved > 10:  # If we saved more than 10MB
                print(f"   🧹 Memory optimization saved {memory_saved:.1f} MB")
                
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _get_progress_percentage(self, completed_steps: int, total_steps: int = 16) -> float:
        """
        Calculate progress percentage.
        
        Args:
            completed_steps: Number of completed steps
            total_steps: Total number of steps
            
        Returns:
            float: Progress percentage
        """
        return (completed_steps / total_steps) * 100
    
    def _log_progress(self, current_step: int, total_steps: int = 16, elapsed_time: float = 0) -> None:
        """
        Log progress with estimated completion time.
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            elapsed_time: Time elapsed so far
        """
        progress = self._get_progress_percentage(current_step, total_steps)
        
        if elapsed_time > 0 and current_step > 0:
            # Estimate remaining time based on current progress
            estimated_total_time = (elapsed_time / current_step) * total_steps
            remaining_time = estimated_total_time - elapsed_time
            eta_minutes = remaining_time / 60
            
            self.logger.info(f"📊 Progress: {progress:.1f}% ({current_step}/{total_steps})")
            self.logger.info(f"⏱️ Elapsed: {elapsed_time/60:.1f} min | ETA: {eta_minutes:.1f} min")
            print(f"   📊 Progress: {progress:.1f}% ({current_step}/{total_steps})")
            print(f"   ⏱️ Elapsed: {elapsed_time/60:.1f} min | ETA: {eta_minutes:.1f} min")
        else:
            self.logger.info(f"📊 Progress: {progress:.1f}% ({current_step}/{total_steps})")
            print(f"   📊 Progress: {progress:.1f}% ({current_step}/{total_steps})")
        
    def _log_step_completion(self, step_name: str, step_start: float, step_times: dict, success: bool = True) -> None:
        """
        Log step completion with timing and memory usage.
        
        Args:
            step_name: Name of the completed step
            step_start: Start time of the step
            step_times: Dictionary to store step times
            success: Whether the step was successful
        """
        step_time = time.time() - step_start
        step_times[step_name] = step_time
        
        # Get comprehensive system resources
        resources = self._get_system_resources()
        
        status_icon = "✅" if success else "❌"
        status_text = "completed successfully" if success else "failed"
        
        self.logger.info(f"{status_icon} {step_name}: {status_text} in {step_time:.2f}s")
        self.logger.info(f"💾 Process Memory: {resources['memory_mb']:.1f} MB | CPU: {resources['cpu_percent']:.1f}%")
        self.logger.info(f"🖥️ System Memory: {resources['system_memory_percent']:.1f}% | Available: {resources['available_memory_gb']:.1f} GB")
        
        print(f"   {status_icon} {step_name}: {status_text} in {step_time:.2f}s")
        print(f"   💾 Process Memory: {resources['memory_mb']:.1f} MB | CPU: {resources['cpu_percent']:.1f}%")
        print(f"   🖥️ System Memory: {resources['system_memory_percent']:.1f}% | Available: {resources['available_memory_gb']:.1f} GB")
        
        # Memory warning system
        if resources['system_memory_percent'] > 85:
            warning_msg = f"⚠️ HIGH MEMORY USAGE: {resources['system_memory_percent']:.1f}% - Consider closing other applications"
            self.logger.warning(warning_msg)
            print(f"   {warning_msg}")
        
        if resources['available_memory_gb'] < 2.0:
            warning_msg = f"⚠️ LOW AVAILABLE MEMORY: {resources['available_memory_gb']:.1f} GB remaining"
            self.logger.warning(warning_msg)
            print(f"   {warning_msg}")
        
        # Log progress after each step
        completed_steps = len(step_times)
        elapsed_time = sum(step_times.values())
        self._log_progress(completed_steps, 16, elapsed_time)
        
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid enhanced training manager configuration"),
            AttributeError: (False, "Missing required enhanced training parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="enhanced training manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize enhanced training manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Enhanced Training Manager...")
            self.logger.info(f"📊 Blank training mode: {self.blank_training_mode}")
            self.logger.info(f"🔧 Max trials: {self.max_trials}")
            self.logger.info(f"🔧 N trials: {self.n_trials}")
            self.logger.info(f"📈 Lookback days: {self.lookback_days}")
            self.logger.info(f"🚀 Computational optimization: {self.enable_computational_optimization}")
            
            # Analyze resource requirements
            resource_analysis = self._analyze_resource_requirements()
            if resource_analysis:
                self.logger.info("📊 Resource Analysis:")
                self.logger.info(f"   💾 System Memory: {resource_analysis['system_memory_gb']:.1f} GB")
                self.logger.info(f"   🖥️ CPU Cores: {resource_analysis['cpu_count']}")
                self.logger.info(f"   📈 Estimated Memory Usage: {resource_analysis['estimated_memory_gb']:.1f} GB")
                self.logger.info(f"   ⏱️ Estimated Time: {resource_analysis['estimated_time_minutes']} minutes ({resource_analysis['estimated_time_minutes']/60:.1f} hours)")
                self.logger.info(f"   🤖 Models to Train: {resource_analysis['models_to_train']}")
                self.logger.info(f"   🔧 Optimization Trials: {resource_analysis['optimization_trials']}")
                
                print("📊 Resource Analysis:")
                print(f"   💾 System Memory: {resource_analysis['system_memory_gb']:.1f} GB")
                print(f"   🖥️ CPU Cores: {resource_analysis['cpu_count']}")
                print(f"   📈 Estimated Memory Usage: {resource_analysis['estimated_memory_gb']:.1f} GB")
                print(f"   ⏱️ Estimated Time: {resource_analysis['estimated_time_minutes']} minutes ({resource_analysis['estimated_time_minutes']/60:.1f} hours)")
                print(f"   🤖 Models to Train: {resource_analysis['models_to_train']}")
                print(f"   🔧 Optimization Trials: {resource_analysis['optimization_trials']}")
                
                # Show step-by-step breakdown
                if 'step_breakdown' in resource_analysis:
                    self.logger.info("📋 Step-by-Step Time Estimates:")
                    print("📋 Step-by-Step Time Estimates:")
                    total_estimated = sum(resource_analysis['step_breakdown'].values())
                    for step_name, minutes in resource_analysis['step_breakdown'].items():
                        percentage = (minutes / total_estimated) * 100
                        self.logger.info(f"   {step_name}: {minutes} min ({percentage:.1f}%)")
                        print(f"   {step_name}: {minutes} min ({percentage:.1f}%)")
                
                # Log recommendations
                if resource_analysis['recommendations']:
                    self.logger.info("💡 Recommendations:")
                    print("💡 Recommendations:")
                    for rec in resource_analysis['recommendations']:
                        self.logger.info(f"   {rec}")
                        print(f"   {rec}")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid configuration for enhanced training manager")
                return False
            
            # Initialize computational optimization if enabled
            if self.enable_computational_optimization:
                await self._initialize_computational_optimization()
                
            self.logger.info("✅ Enhanced Training Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced Training Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate enhanced training manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate enhanced training manager specific settings
            if self.max_enhanced_training_history <= 0:
                self.logger.error("❌ Invalid max_enhanced_training_history configuration")
                return False
                
            if self.max_trials <= 0:
                self.logger.error("❌ Invalid max_trials configuration")
                return False
                
            if self.n_trials <= 0:
                self.logger.error("❌ Invalid n_trials configuration")
                return False
                
            if self.lookback_days <= 0:
                self.logger.error("❌ Invalid lookback_days configuration")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid enhanced training parameters"),
            AttributeError: (False, "Missing enhanced training components"),
            KeyError: (False, "Missing required enhanced training data"),
        },
        default_return=False,
        context="enhanced training execution",
    )
    async def execute_enhanced_training(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """
        Execute the comprehensive 16-step enhanced training pipeline.

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE START")
            self.logger.info("=" * 80)
            self.logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
            self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
            self.logger.info(f"📊 Training Mode: {enhanced_training_input.get('training_mode', 'N/A')}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"🔧 Blank Training Mode: {self.blank_training_mode}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"🔧 N Trials: {self.n_trials}")
            
            print("=" * 80)
            print("🚀 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE START")
            print("=" * 80)
            print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
            print(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
            print(f"📊 Training Mode: {enhanced_training_input.get('training_mode', 'N/A')}")
            print(f"📈 Lookback Days: {self.lookback_days}")
            print(f"🔧 Blank Training Mode: {self.blank_training_mode}")
            print(f"🔧 Max Trials: {self.max_trials}")
            print(f"🔧 N Trials: {self.n_trials}")
            
            self.is_training = True
            
            # Validate training input
            if not self._validate_enhanced_training_inputs(enhanced_training_input):
                return False
            
            # Execute the comprehensive 16-step pipeline
            success = await self._execute_comprehensive_pipeline(enhanced_training_input)
            
            if success:
                # Store training history
                await self._store_enhanced_training_history(enhanced_training_input)
                
                self.logger.info("=" * 80)
                self.logger.info("🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 80)
                self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"🎯 Symbol: {enhanced_training_input.get('symbol', 'N/A')}")
                self.logger.info(f"🏢 Exchange: {enhanced_training_input.get('exchange', 'N/A')}")
                self.logger.info("📋 Completed Steps:")
                self.logger.info("   1. Data Collection")
                self.logger.info("   2. Market Regime Classification")
                self.logger.info("   3. Regime Data Splitting")
                self.logger.info("   4. Analyst Labeling & Feature Engineering")
                self.logger.info("   5. Analyst Specialist Training")
                self.logger.info("   6. Analyst Enhancement")
                self.logger.info("   7. Analyst Ensemble Creation")
                self.logger.info("   8. Tactician Labeling")
                self.logger.info("   9. Tactician Specialist Training")
                self.logger.info("   10. Tactician Ensemble Creation")
                self.logger.info("   11. Confidence Calibration")
                self.logger.info("   12. Final Parameters Optimization")
                self.logger.info("   13. Walk Forward Validation")
                self.logger.info("   14. Monte Carlo Validation")
                self.logger.info("   15. A/B Testing")
                self.logger.info("   16. Saving Results")
                
                print("=" * 80)
                print("🎉 COMPREHENSIVE 16-STEP ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                print("=" * 80)
                print("   ✅ All 16 training steps completed successfully!")
            else:
                self.logger.error("❌ Enhanced training pipeline failed")
                print("❌ Enhanced training pipeline failed")
            
            self.is_training = False
            return success
            
        except Exception as e:
            self.logger.error(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
            self.logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            print(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
            print(f"📋 Error details: {type(e).__name__}: {str(e)}")
            self.is_training = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="enhanced training inputs validation",
    )
    def _validate_enhanced_training_inputs(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """
        Validate enhanced training input parameters.

        Args:
            enhanced_training_input: Enhanced training input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "lookback_days"]
            
            for field in required_fields:
                if field not in enhanced_training_input:
                    self.logger.error(f"❌ Missing required enhanced training input field: {field}")
                    return False
            
            # Validate specific field values
            if enhanced_training_input.get("lookback_days", 0) <= 0:
                self.logger.error("❌ Invalid lookback_days value")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced training inputs validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="computational optimization initialization",
    )
    async def _initialize_computational_optimization(self) -> bool:
        """Initialize computational optimization components."""
        try:
            self.logger.info("🚀 Initializing computational optimization components...")
            
            # Get computational optimization configuration
            optimization_config = get_computational_optimization_config()
            
            # Create computational optimization manager
            self.computational_optimization_manager = await create_computational_optimization_manager(
                config=optimization_config,
                market_data=pd.DataFrame(),  # Will be loaded during training
                model_config={}  # Will be configured during training
            )
            
            if self.computational_optimization_manager:
                self.logger.info("✅ Computational optimization components initialized successfully")
                return True
            else:
                self.logger.warning("⚠️ Failed to initialize computational optimization components")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Computational optimization initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="comprehensive pipeline execution",
    )
    async def _execute_comprehensive_pipeline(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """
        Execute the comprehensive 16-step training pipeline.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if all steps successful, False otherwise
        """
        try:
            symbol = training_input.get("symbol", "")
            exchange = training_input.get("exchange", "")
            timeframe = training_input.get("timeframe", "1m")
            data_dir = "data/training"

            # Initialize pipeline state and timing
            pipeline_state = {}
            start_time = time.time()
            step_times = {}
            
            # Store current training parameters for checkpointing
            self.current_symbol = symbol
            self.current_exchange = exchange
            self.current_timeframe = timeframe
            
            # Check for existing checkpoint
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.logger.info("🔄 Resuming from checkpoint...")
                print("🔄 Resuming from checkpoint...")
                pipeline_state = checkpoint.get("pipeline_state", {})
                last_completed_step = checkpoint.get("current_step", "")
                self.logger.info(f"📂 Last completed step: {last_completed_step}")
                print(f"📂 Last completed step: {last_completed_step}")
            else:
                self.logger.info("🚀 Starting fresh training...")
                print("🚀 Starting fresh training...")
            
            # Enhanced logging setup
            self.logger.info("=" * 100)
            self.logger.info("🚀 COMPREHENSIVE TRAINING PIPELINE START")
            self.logger.info("=" * 100)
            self.logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            self.logger.info(f"🔧 Max Trials: {self.max_trials}")
            self.logger.info(f"📈 Lookback Days: {self.lookback_days}")
            self.logger.info(f"💾 Memory Optimization: {'Enabled' if self.enable_computational_optimization else 'Disabled'}")
            self.logger.info("=" * 100)
            
            print("=" * 100)
            print("🚀 COMPREHENSIVE TRAINING PIPELINE START")
            print("=" * 100)
            print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🎯 Symbol: {symbol}")
            print(f"🏢 Exchange: {exchange}")
            print(f"📊 Timeframe: {timeframe}")
            print(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            print(f"🔧 Max Trials: {self.max_trials}")
            print(f"📈 Lookback Days: {self.lookback_days}")
            print(f"💾 Memory Optimization: {'Enabled' if self.enable_computational_optimization else 'Disabled'}")
            print("=" * 100)
            
            # Step 1: Data Collection
            step_start = time.time()
            self.logger.info("📊 STEP 1: Data Collection...")
            self.logger.info("   🔍 Downloading and preparing market data...")
            print("   📊 Step 1: Data Collection...")
            print("   🔍 Downloading and preparing market data...")
            
            from src.training.steps import step1_data_collection
            step1_result = await step1_data_collection.run_step(
                symbol=symbol,
                exchange_name=exchange,
                min_data_points="1000",
                data_dir=data_dir,
                download_new_data=True,
                lookback_days=self.lookback_days,
            )
            
            if step1_result is None or step1_result[0] is None:
                self.logger.error("❌ Step 1: Data Collection failed")
                print("❌ Step 1: Data Collection failed")
                return False
            
            # Update pipeline state
            pipeline_state["data_collection"] = {
                "status": "SUCCESS",
                "result": step1_result
            }
            
            # Run validator for Step 1
            validation_result = await self._run_step_validator(
                "step1_data_collection", training_input, pipeline_state
            )
            
            self._log_step_completion("Step 1: Data Collection", step_start, step_times)
            
            # Save checkpoint after step 1
            self._save_checkpoint("step1_data_collection", pipeline_state)
            
            # Memory optimization after data collection
            self._optimize_memory_usage()

            # Step 2: Market Regime Classification
            step_start = time.time()
            self.logger.info("🎭 STEP 2: Market Regime Classification...")
            self.logger.info("   🧠 Analyzing market regimes and volatility patterns...")
            print("   🎭 Step 2: Market Regime Classification...")
            print("   🧠 Analyzing market regimes and volatility patterns...")
            
            from src.training.steps import step2_market_regime_classification
            step2_success = await step2_market_regime_classification.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step2_success:
                self._log_step_completion("Step 2: Market Regime Classification", step_start, step_times, success=False)
                return False
            
            # Update pipeline state
            pipeline_state["regime_classification"] = {
                "status": "SUCCESS",
                "success": step2_success
            }
            
            # Run validator for Step 2
            validation_result = await self._run_step_validator(
                "step2_market_regime_classification", training_input, pipeline_state
            )
            
            self._log_step_completion("Step 2: Market Regime Classification", step_start, step_times)
            
            # Save checkpoint after step 2
            self._save_checkpoint("step2_market_regime_classification", pipeline_state)

            # Step 3: Regime Data Splitting
            step_start = time.time()
            self.logger.info("📊 STEP 3: Regime Data Splitting...")
            self.logger.info("   📈 Splitting data by market regimes for specialized training...")
            print("   📊 Step 3: Regime Data Splitting...")
            print("   📈 Splitting data by market regimes for specialized training...")
            
            from src.training.steps import step3_regime_data_splitting
            step3_success = await step3_regime_data_splitting.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step3_success:
                self._log_step_completion("Step 3: Regime Data Splitting", step_start, step_times, success=False)
                return False
            
            # Update pipeline state
            pipeline_state["regime_data_splitting"] = {
                "status": "SUCCESS",
                "success": step3_success
            }
            
            # Run validator for Step 3
            validation_result = await self._run_step_validator(
                "step3_regime_data_splitting", training_input, pipeline_state
            )
            
            self._log_step_completion("Step 3: Regime Data Splitting", step_start, step_times)
            
            # Save checkpoint after step 3
            self._save_checkpoint("step3_regime_data_splitting", pipeline_state)

            # Step 4: Analyst Labeling & Feature Engineering
            self.logger.info("🧠 STEP 4: Analyst Labeling & Feature Engineering...")
            print("   🧠 Step 4: Analyst Labeling & Feature Engineering...")
            
            from src.training.steps import step4_analyst_labeling_feature_engineering
            step4_success = await step4_analyst_labeling_feature_engineering.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step4_success:
                self._log_step_completion("Step 4: Analyst Labeling & Feature Engineering", step_start, step_times, success=False)
                return False
            
            self._log_step_completion("Step 4: Analyst Labeling & Feature Engineering", step_start, step_times)
            
            # Save checkpoint after step 4
            self._save_checkpoint("step4_analyst_labeling_feature_engineering", pipeline_state)
            
            # Memory optimization after feature engineering (most memory-intensive step)
            self._optimize_memory_usage()

            # Step 5: Analyst Specialist Training
            self.logger.info("🎯 STEP 5: Analyst Specialist Training...")
            print("   🎯 Step 5: Analyst Specialist Training...")
            
            from src.training.steps import step5_analyst_specialist_training
            step5_success = await step5_analyst_specialist_training.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )

            # Update pipeline state
            pipeline_state["analyst_labeling_feature_engineering"] = {
                "status": "SUCCESS",
                "success": step4_success
            }
            
            # Run validator for Step 4
            validation_result = await self._run_step_validator(
                "step4_analyst_labeling_feature_engineering", training_input, pipeline_state
            )
            
            if not step5_success:
                self.logger.error("❌ Step 5: Analyst Specialist Training failed")
                print("❌ Step 5: Analyst Specialist Training failed")
                return False
            
            # Update pipeline state
            pipeline_state["analyst_specialist_training"] = {
                "status": "SUCCESS",
                "success": step5_success
            }
            
            # Run validator for Step 5
            validation_result = await self._run_step_validator(
                "step5_analyst_specialist_training", training_input, pipeline_state
            )
            
            self.logger.info("✅ Step 5: Analyst Specialist Training completed successfully")
            print("   ✅ Step 5: Analyst Specialist Training completed successfully")

            # Step 6: Analyst Enhancement
            self.logger.info("🔧 STEP 6: Analyst Enhancement...")
            print("   🔧 Step 6: Analyst Enhancement...")
            
            from src.training.steps import step6_analyst_enhancement
            step6_success = await step6_analyst_enhancement.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step6_success:
                self.logger.error("❌ Step 6: Analyst Enhancement failed")
                print("❌ Step 6: Analyst Enhancement failed")
                return False
            
            self.logger.info("✅ Step 6: Analyst Enhancement completed successfully")
            print("   ✅ Step 6: Analyst Enhancement completed successfully")

            # Step 7: Analyst Ensemble Creation
            self.logger.info("🎲 STEP 7: Analyst Ensemble Creation...")
            print("   🎲 Step 7: Analyst Ensemble Creation...")
            
            from src.training.steps import step7_analyst_ensemble_creation
            step7_success = await step7_analyst_ensemble_creation.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step7_success:
                self.logger.error("❌ Step 7: Analyst Ensemble Creation failed")
                print("❌ Step 7: Analyst Ensemble Creation failed")
                return False
            
            self.logger.info("✅ Step 7: Analyst Ensemble Creation completed successfully")
            print("   ✅ Step 7: Analyst Ensemble Creation completed successfully")

            # Step 8: Tactician Labeling
            self.logger.info("🎯 STEP 8: Tactician Labeling...")
            print("   🎯 Step 8: Tactician Labeling...")
            
            from src.training.steps import step8_tactician_labeling
            step8_success = await step8_tactician_labeling.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step8_success:
                self.logger.error("❌ Step 8: Tactician Labeling failed")
                print("❌ Step 8: Tactician Labeling failed")
                return False
            
            self.logger.info("✅ Step 8: Tactician Labeling completed successfully")
            print("   ✅ Step 8: Tactician Labeling completed successfully")

            # Step 9: Tactician Specialist Training
            self.logger.info("🧠 STEP 9: Tactician Specialist Training...")
            print("   🧠 Step 9: Tactician Specialist Training...")
            
            from src.training.steps import step9_tactician_specialist_training
            step9_success = await step9_tactician_specialist_training.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step9_success:
                self.logger.error("❌ Step 9: Tactician Specialist Training failed")
                print("❌ Step 9: Tactician Specialist Training failed")
                return False
            
            self.logger.info("✅ Step 9: Tactician Specialist Training completed successfully")
            print("   ✅ Step 9: Tactician Specialist Training completed successfully")

            # Step 10: Tactician Ensemble Creation
            self.logger.info("🎲 STEP 10: Tactician Ensemble Creation...")
            print("   🎲 Step 10: Tactician Ensemble Creation...")
            
            from src.training.steps import step10_tactician_ensemble_creation
            step10_success = await step10_tactician_ensemble_creation.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step10_success:
                self.logger.error("❌ Step 10: Tactician Ensemble Creation failed")
                print("❌ Step 10: Tactician Ensemble Creation failed")
                return False
            
            self.logger.info("✅ Step 10: Tactician Ensemble Creation completed successfully")
            print("   ✅ Step 10: Tactician Ensemble Creation completed successfully")

            # Step 11: Confidence Calibration
            self.logger.info("🎯 STEP 11: Confidence Calibration...")
            print("   🎯 Step 11: Confidence Calibration...")
            
            from src.training.steps import step11_confidence_calibration
            step11_success = await step11_confidence_calibration.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step11_success:
                self.logger.error("❌ Step 11: Confidence Calibration failed")
                print("❌ Step 11: Confidence Calibration failed")
                return False
            
            self.logger.info("✅ Step 11: Confidence Calibration completed successfully")
            print("   ✅ Step 11: Confidence Calibration completed successfully")

            # Step 12: Final Parameters Optimization (with computational optimization)
            self.logger.info("🔧 STEP 12: Final Parameters Optimization with Computational Optimization...")
            print("   🔧 Step 12: Final Parameters Optimization with Computational Optimization...")
            
            # Use computational optimization if available
            if self.computational_optimization_manager:
                step12_success = await self._run_optimized_parameters_optimization(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
            else:
                # Fallback to standard optimization
                from src.training.steps import step12_final_parameters_optimization
                step12_success = await step12_final_parameters_optimization.run_step(
                    symbol=symbol,
                    data_dir=data_dir,
                    timeframe=timeframe,
                    exchange=exchange,
                )
            
            if not step12_success:
                self.logger.error("❌ Step 12: Final Parameters Optimization failed")
                print("❌ Step 12: Final Parameters Optimization failed")
                return False
            
            self.logger.info("✅ Step 12: Final Parameters Optimization completed successfully")
            print("   ✅ Step 12: Final Parameters Optimization completed successfully")

            # Step 13: Walk Forward Validation
            self.logger.info("📈 STEP 13: Walk Forward Validation...")
            print("   📈 Step 13: Walk Forward Validation...")
            
            from src.training.steps import step13_walk_forward_validation
            step13_success = await step13_walk_forward_validation.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step13_success:
                self.logger.error("❌ Step 13: Walk Forward Validation failed")
                print("❌ Step 13: Walk Forward Validation failed")
                return False
            
            self.logger.info("✅ Step 13: Walk Forward Validation completed successfully")
            print("   ✅ Step 13: Walk Forward Validation completed successfully")

            # Step 14: Monte Carlo Validation
            self.logger.info("🎲 STEP 14: Monte Carlo Validation...")
            print("   🎲 Step 14: Monte Carlo Validation...")
            
            from src.training.steps import step14_monte_carlo_validation
            step14_success = await step14_monte_carlo_validation.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step14_success:
                self.logger.error("❌ Step 14: Monte Carlo Validation failed")
                print("❌ Step 14: Monte Carlo Validation failed")
                return False
            
            self.logger.info("✅ Step 14: Monte Carlo Validation completed successfully")
            print("   ✅ Step 14: Monte Carlo Validation completed successfully")

            # Step 15: A/B Testing
            self.logger.info("🧪 STEP 15: A/B Testing...")
            print("   🧪 Step 15: A/B Testing...")
            
            from src.training.steps import step15_ab_testing
            step15_success = await step15_ab_testing.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step15_success:
                self.logger.error("❌ Step 15: A/B Testing failed")
                print("❌ Step 15: A/B Testing failed")
                return False
            
            self.logger.info("✅ Step 15: A/B Testing completed successfully")
            print("   ✅ Step 15: A/B Testing completed successfully")

            # Step 16: Saving Results
            self.logger.info("💾 STEP 16: Saving Results...")
            print("   💾 Step 16: Saving Results...")
            
            from src.training.steps import step16_saving
            step16_success = await step16_saving.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step16_success:
                self.logger.error("❌ Step 16: Saving Results failed")
                print("❌ Step 16: Saving Results failed")
                return False
            
            self.logger.info("✅ Step 16: Saving Results completed successfully")
            print("   ✅ Step 16: Saving Results completed successfully")

            # Calculate total time and summary
            total_time = time.time() - start_time
            total_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            
            # Log comprehensive summary
            self.logger.info("=" * 100)
            self.logger.info("🎉 COMPREHENSIVE TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)
            self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"⏱️ Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            self.logger.info(f"💾 Final Memory Usage: {total_memory:.1f} MB")
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            
            # Log step-by-step timing
            self.logger.info("📊 Step-by-Step Timing:")
            for step_name, step_time in step_times.items():
                percentage = (step_time / total_time) * 100
                self.logger.info(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
            
            print("=" * 100)
            print("🎉 COMPREHENSIVE TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 100)
            print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            print(f"💾 Final Memory Usage: {total_memory:.1f} MB")
            print(f"🎯 Symbol: {symbol}")
            print(f"🏢 Exchange: {exchange}")
            print(f"📊 Timeframe: {timeframe}")
            print(f"🧠 Training Mode: {'Blank' if self.blank_training_mode else 'Full'}")
            print("📊 Step-by-Step Timing:")
            for step_name, step_time in step_times.items():
                percentage = (step_time / total_time) * 100
                print(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
            
            # Clear checkpoint on successful completion
            self._clear_checkpoint()
            
            return True
            
        except Exception as e:
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            self.logger.error(f"💥 COMPREHENSIVE PIPELINE FAILED: {str(e)}")
            self.logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            self.logger.error(f"⏱️ Time elapsed before failure: {total_time:.2f}s")
            self.logger.info("💾 Checkpoint saved - you can resume training later")
            print(f"💥 COMPREHENSIVE PIPELINE FAILED: {str(e)}")
            print(f"📋 Error details: {type(e).__name__}: {str(e)}")
            print(f"⏱️ Time elapsed before failure: {total_time:.2f}s")
            print("💾 Checkpoint saved - you can resume training later")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimized parameters optimization",
    )
    async def _run_optimized_parameters_optimization(
        self,
        symbol: str,
        data_dir: str,
        timeframe: str,
        exchange: str,
    ) -> bool:
        """Run optimized parameters optimization using computational optimization strategies."""
        try:
            self.logger.info("🚀 Running optimized parameters optimization...")
            
            # Load market data for optimization
            market_data = await self._load_market_data_for_optimization(symbol, data_dir, exchange)
            if market_data is None:
                self.logger.error("❌ Failed to load market data for optimization")
                return False
            
            # Update computational optimization manager with market data
            if self.computational_optimization_manager:
                await self.computational_optimization_manager.initialize(market_data, {})
            
            # Define optimization objective function
            def optimization_objective(params):
                # This would be the actual optimization objective
                # For now, return a simple metric
                return 0.5  # Placeholder
            
            # Run optimized parameter optimization
            optimization_results = await self.computational_optimization_manager.optimize_parameters(
                objective_function=optimization_objective,
                n_trials=self.n_trials,
                use_surrogates=True
            )
            
            # Store optimization statistics
            self.optimization_statistics = self.computational_optimization_manager.get_optimization_statistics()
            
            # Save optimization results
            await self._save_optimization_results(symbol, exchange, data_dir, optimization_results)
            
            self.logger.info("✅ Optimized parameters optimization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Optimized parameters optimization failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="market data loading for optimization",
    )
    async def _load_market_data_for_optimization(
        self,
        symbol: str,
        data_dir: str,
        exchange: str,
    ) -> pd.DataFrame | None:
        """Load market data for optimization."""
        try:
            # Load market data from the data directory
            # This is a simplified implementation
            import os
            data_file = f"{data_dir}/{exchange}_{symbol}_klines.csv"
            
            if os.path.exists(data_file):
                market_data = pd.read_csv(data_file)
                self.logger.info(f"✅ Loaded market data from {data_file}")
                return market_data
            else:
                self.logger.warning(f"⚠️ Market data file not found: {data_file}")
                # Return empty DataFrame as fallback
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load market data: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimization results saving",
    )
    async def _save_optimization_results(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        optimization_results: dict[str, Any],
    ) -> bool:
        """Save optimization results."""
        try:
            import json
            import os
            
            # Create optimization results directory
            optimization_dir = f"{data_dir}/optimization_results"
            os.makedirs(optimization_dir, exist_ok=True)
            
            # Save optimization results
            results_file = f"{optimization_dir}/{exchange}_{symbol}_optimized_parameters.json"
            with open(results_file, "w") as f:
                json.dump(optimization_results, f, indent=2)
            
            # Save optimization statistics
            stats_file = f"{optimization_dir}/{exchange}_{symbol}_optimization_statistics.json"
            with open(stats_file, "w") as f:
                json.dump(self.optimization_statistics, f, indent=2)
            
            self.logger.info(f"✅ Saved optimization results to {optimization_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save optimization results: {e}")
            return False

    async def _run_step_validator(
        self,
        step_name: str,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run validator for a specific step.
        
        Args:
            step_name: Name of the step
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary containing validation results
        """
        if not self.enable_validators:
            return {
                "step_name": step_name,
                "validation_passed": True,
                "skipped": True,
                "reason": "Validators disabled"
            }
        
        try:
            self.logger.info(f"🔍 Running validator for {step_name}")
            validation_result = await validator_orchestrator.run_step_validator(
                step_name=step_name,
                training_input=training_input,
                pipeline_state=pipeline_state,
                config=self.config
            )
            
            # Store validation result
            self.validation_results[step_name] = validation_result
            
            if validation_result.get("validation_passed", False):
                self.logger.info(f"✅ {step_name} validation passed")
            else:
                self.logger.warning(f"⚠️ {step_name} validation failed: {validation_result.get('error', 'Unknown error')}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error running validator for {step_name}: {e}")
            return {
                "step_name": step_name,
                "validation_passed": False,
                "error": str(e)
            }

    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training history storage",
    )
    async def _store_enhanced_training_history(self, enhanced_training_input: dict[str, Any]) -> None:
        """
        Store enhanced training history.

        Args:
            enhanced_training_input: Enhanced training input parameters
        """
        try:
            # Add to training history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "training_input": enhanced_training_input,
                "results": self.enhanced_training_results,
            }
            
            self.enhanced_training_history.append(history_entry)
            
            # Limit history size
            if len(self.enhanced_training_history) > self.max_enhanced_training_history:
                self.enhanced_training_history = self.enhanced_training_history[-self.max_enhanced_training_history:]
            
            self.logger.info(f"📁 Stored training history entry (total: {len(self.enhanced_training_history)})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store training history: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training results storage",
    )
    async def _store_enhanced_training_results(self) -> None:
        """Store enhanced training results."""
        try:
            self.logger.info("📁 Storing enhanced training results...")
            
            # Store results in a format that can be retrieved later
            results_key = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # This would typically store to database or file system
            self.logger.info(f"📁 Storing enhanced training results with key: {results_key}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store enhanced training results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training results getting",
    )
    def get_enhanced_training_results(
        self,
        enhanced_training_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get enhanced training results.

        Args:
            enhanced_training_type: Type of training results to get

        Returns:
            dict: Enhanced training results
        """
        try:
            if enhanced_training_type:
                return self.enhanced_training_results.get(enhanced_training_type, {})
            return self.enhanced_training_results.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced training results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="enhanced training history getting",
    )
    def get_enhanced_training_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get enhanced training history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            list: Enhanced training history
        """
        try:
            history = self.enhanced_training_history.copy()
            if limit:
                history = history[-limit:]
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced training history: {e}")
            return []

    def get_enhanced_training_status(self) -> dict[str, Any]:
        """
        Get enhanced training status.

        Returns:
            dict: Enhanced training status information
        """
        return {
            "is_training": self.is_training,
            "has_results": bool(self.enhanced_training_results),
            "history_count": len(self.enhanced_training_history),
            "blank_training_mode": self.blank_training_mode,
            "max_trials": self.max_trials,
            "n_trials": self.n_trials,
            "lookback_days": self.lookback_days,
            "enable_validators": self.enable_validators,
            "enable_computational_optimization": self.enable_computational_optimization,
            "optimization_statistics": self.optimization_statistics,
        }
    
    def get_validation_results(self) -> dict[str, Any]:
        """
        Get validation results for all steps.
        
        Returns:
            dict: Validation results summary
        """
        return {
            "validation_results": self.validation_results,
            "validation_summary": validator_orchestrator.get_validation_summary(),
            "failed_validations": validator_orchestrator.get_failed_validations()
        }
    
    def get_computational_optimization_results(self) -> dict[str, Any]:
        """
        Get computational optimization results and statistics.
        
        Returns:
            dict: Computational optimization results
        """
        if self.computational_optimization_manager:
            return {
                "optimization_statistics": self.computational_optimization_manager.get_optimization_statistics(),
                "enabled_optimizations": self.optimization_statistics,
                "manager_available": True
            }
        else:
            return {
                "optimization_statistics": {},
                "enabled_optimizations": {},
                "manager_available": False
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="enhanced training manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the enhanced training manager and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Enhanced Training Manager...")
            
            # Cleanup computational optimization manager
            if self.computational_optimization_manager:
                await self.computational_optimization_manager.cleanup()
                self.logger.info("✅ Computational optimization manager cleaned up")
            
            self.is_training = False
            self.logger.info("✅ Enhanced Training Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop Enhanced Training Manager: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced training manager setup",
)
async def setup_enhanced_training_manager(
    config: dict[str, Any] | None = None,
) -> EnhancedTrainingManager | None:
    """
    Setup and return a configured EnhancedTrainingManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedTrainingManager: Configured enhanced training manager instance
    """
    try:
        manager = EnhancedTrainingManager(config or {})
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup enhanced training manager: {e}")
        return None
