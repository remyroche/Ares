# src/training/launcher_integration_patch.py

"""
Integration patch for ares_launcher.py to enable the new optimization features.
This module contains the updated training functions that use the optimized training manager.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from src.config.computational_optimization_config import get_optimization_config
from src.training.enhanced_training_manager_optimized import EnhancedTrainingManagerOptimized
from src.training.factory import OptimizedTrainingFactory
from src.training.memory_profiler import MemoryProfiler, MemoryLeakDetector
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class OptimizedAresLauncherMixin:
    """
    Mixin class that provides optimized training methods for AresLauncher.
    This can be mixed into the existing AresLauncher class to add optimization features.
    """
    
    def __init__(self):
        # Initialize optimization components
        self.optimization_enabled = True
        self.memory_profiler = None
        self.leak_detector = None
        self.optimization_factory = None
        
    def _setup_optimization_components(self, config: Dict[str, Any]):
        """Setup optimization components if enabled."""
        if not self.optimization_enabled:
            return
            
        try:
            # Create optimization factory
            self.optimization_factory = OptimizedTrainingFactory(config)
            
            # Create memory profiler
            self.memory_profiler = self.optimization_factory.create_memory_profiler()
            
            # Create leak detector
            if self.memory_profiler:
                self.leak_detector = self.optimization_factory.create_memory_leak_detector(
                    self.memory_profiler
                )
            
            self.logger.info("✅ Optimization components initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup optimization components: {e}")
            self.optimization_enabled = False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimized_training_pipeline",
    )
    def _run_optimized_unified_training(
        self,
        symbol: str,
        exchange: str,
        training_mode: str,
        lookback_days: int,
        with_gui: bool = False,
    ):
        """Run optimized unified training with enhanced training manager."""
        # Set environment variable for blank training mode
        if training_mode == "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            print("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1")

        mode_display = f"{training_mode} training (OPTIMIZED)"
        print(f"🚀 Starting {mode_display} for {symbol} on {exchange}")
        self.logger.info(f"🚀 Starting {mode_display} for {symbol} on {exchange}")

        @handle_errors(
            exceptions=(Exception,),
            default_return=False,
            context="optimized_enhanced_training_pipeline",
        )
        async def run_optimized_enhanced_training():
            """Execute optimized enhanced training using EnhancedTrainingManagerOptimized."""
            from src.database.sqlite_manager import SQLiteManager

            logger = system_logger.getChild("OptimizedEnhancedTrainingPipeline")
            
            logger.info("=" * 80)
            logger.info("🚀 OPTIMIZED ENHANCED TRAINING PIPELINE START")
            logger.info("=" * 80)
            logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🎯 Symbol: {symbol}")
            logger.info(f"🏢 Exchange: {exchange}")
            logger.info(f"📊 Training Mode: {training_mode}")
            logger.info(f"📈 Lookback Days: {lookback_days}")
            logger.info("🔧 OPTIMIZATIONS ENABLED")
            print("=" * 80)
            print("🚀 OPTIMIZED ENHANCED TRAINING PIPELINE START")
            print("=" * 80)

            try:
                # Initialize database manager
                logger.info("📊 STEP 0: Initializing Database Manager...")
                print("   📊 Setting up database manager...")
                
                default_config = {
                    "database": {
                        "sqlite_path": "data/ares.db",
                        "backup_enabled": True,
                        "max_connections": 10,
                        "timeout": 30,
                        "check_same_thread": False,
                    },
                }
                
                db_manager = SQLiteManager(default_config, exchange=exchange, symbol=symbol)
                await db_manager.initialize()
                logger.info("✅ Database manager initialized successfully")
                print("   ✅ Database manager initialized successfully")

                # Setup optimization configuration
                logger.info("🔧 STEP 1: Setting up optimization configuration...")
                print("   🔧 Setting up optimization configuration...")
                
                # Get optimization config with custom settings for training mode
                custom_optimization = {}
                
                if training_mode == "blank":
                    # More aggressive optimizations for blank mode
                    custom_optimization = {
                        "parallelization": {"max_workers": min(os.cpu_count(), 6)},
                        "memory_management": {"memory_threshold": 0.75},
                        "early_stopping": {"patience": 5, "min_trials": 10},
                        "caching": {"max_cache_size": 500}
                    }
                else:
                    # Conservative optimizations for full training
                    custom_optimization = {
                        "parallelization": {"max_workers": min(os.cpu_count(), 8)},
                        "memory_management": {"memory_threshold": 0.8},
                        "early_stopping": {"patience": 10, "min_trials": 20},
                        "caching": {"max_cache_size": 1000}
                    }
                
                optimization_config = get_optimization_config(custom_optimization)
                
                training_config = {
                    "enhanced_training_manager": {
                        "enhanced_training_interval": 3600,
                        "max_enhanced_training_history": 100,
                        "enable_advanced_model_training": True,
                        "enable_ensemble_training": True,
                        "enable_multi_timeframe_training": True,
                        "enable_adaptive_training": True,
                        # Optimized parameters for different training modes
                        "blank_training_mode": training_mode == "blank",
                        "max_trials": 5 if training_mode == "blank" else 200,
                        "n_trials": 8 if training_mode == "blank" else 100,
                        "epochs": 20 if training_mode == "blank" else 100,
                        "batch_size": 128 if training_mode == "blank" else 64,
                        "lookback_days": lookback_days,
                        "enable_all_functions": True,
                    },
                    "computational_optimization": optimization_config,
                    "database": default_config["database"],
                }

                # Initialize optimized enhanced training manager
                logger.info("🤖 STEP 2: Initializing Optimized Enhanced Training Manager...")
                print("   🤖 Initializing optimized enhanced training manager...")
                
                training_manager = EnhancedTrainingManagerOptimized(training_config)
                
                # Initialize optimization components
                if hasattr(self, '_setup_optimization_components'):
                    self._setup_optimization_components(training_config)
                
                # Initialize the training manager
                if not await training_manager.initialize():
                    logger.error("❌ Failed to initialize optimized enhanced training manager")
                    print("❌ Failed to initialize optimized enhanced training manager")
                    return False
                    
                logger.info("✅ Optimized enhanced training manager initialized successfully")
                print("   ✅ Optimized enhanced training manager initialized successfully")
                
                # Take initial memory snapshot
                if self.memory_profiler:
                    initial_snapshot = self.memory_profiler.take_snapshot("training_start")
                    logger.info(f"📊 Initial memory usage: {initial_snapshot['process_memory']['rss_mb']:.1f}MB")

                # Execute the optimized enhanced training
                logger.info("🚀 STEP 3: Executing Optimized Enhanced Training Pipeline...")
                print("   🚀 Starting optimized enhanced training pipeline...")

                # Execute optimized training
                success = await training_manager.execute_optimized_training(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe="1h"
                )

                # Check for memory leaks
                if self.leak_detector:
                    leak_results = self.leak_detector.check_for_leaks()
                    if leak_results["leak_detected"]:
                        logger.warning("⚠️ Memory leak detected during training!")
                        for indicator in leak_results.get("indicators", []):
                            logger.warning(f"  - {indicator}")
                    else:
                        logger.info("✅ No memory leaks detected")

                # Get optimization statistics
                optimization_stats = training_manager.get_optimization_stats()
                logger.info(f"📊 Optimization Statistics: {optimization_stats}")
                
                # Take final memory snapshot
                if self.memory_profiler:
                    final_snapshot = self.memory_profiler.take_snapshot("training_end")
                    memory_usage = final_snapshot['process_memory']['rss_mb']
                    logger.info(f"📊 Final memory usage: {memory_usage:.1f}MB")

                if success:
                    logger.info("=" * 80)
                    logger.info("🎉 OPTIMIZED ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                    logger.info("=" * 80)
                    logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"🎯 Symbol: {symbol}")
                    logger.info(f"🏢 Exchange: {exchange}")
                    logger.info(f"📊 Training Mode: {training_mode}")
                    logger.info("🔧 WITH OPTIMIZATIONS")
                    print("=" * 80)
                    print("🎉 OPTIMIZED ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                    print("=" * 80)
                    print("   ✅ Optimized enhanced training completed successfully!")
                    
                    # Print optimization summary
                    if optimization_stats:
                        print("   📊 Optimization Summary:")
                        for key, value in optimization_stats.items():
                            if isinstance(value, bool):
                                status = "✅" if value else "❌"
                                print(f"     {key}: {status}")
                            else:
                                print(f"     {key}: {value}")
                    
                    return True
                else:
                    logger.error("❌ Optimized enhanced training pipeline failed")
                    print("❌ Optimized enhanced training pipeline failed")
                    return False

            except Exception as e:
                logger.error(f"💥 OPTIMIZED ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
                logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
                print(f"💥 OPTIMIZED ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
                print(f"📋 Error details: {type(e).__name__}: {str(e)}")
                return False

            finally:
                # Cleanup
                try:
                    if 'training_manager' in locals():
                        await training_manager.cleanup()
                        logger.info("🧹 Optimized training manager cleaned up successfully")
                    
                    if 'db_manager' in locals():
                        await db_manager.stop()
                        logger.info("🧹 Database manager cleaned up successfully")
                        
                    if self.memory_profiler:
                        self.memory_profiler.stop_continuous_monitoring()
                        logger.info("🧹 Memory profiler stopped")
                        
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")

        # Run the async optimized training
        print("🔄 Starting optimized async training execution...")
        print("⏳ Optimized training is running... This should be faster than before!")
        print("📊 You can monitor progress in the logs directory.")

        success = asyncio.run(run_optimized_enhanced_training())

        if success:
            self.logger.info(f"✅ {mode_display} completed successfully")
            print(f"✅ {mode_display} completed successfully")
            print("🎉 Optimized training pipeline finished!")
            return True
        self.logger.error(f"❌ {mode_display} failed")
        print(f"❌ {mode_display} failed")
        print("💥 Optimized training pipeline encountered an error.")
        return False

    def run_optimized_enhanced_blank_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run optimized enhanced blank training."""
        return self._run_optimized_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="blank",
            lookback_days=60,
            with_gui=with_gui,
        )

    def run_optimized_backtesting(
        self, 
        symbol: str, 
        exchange: str, 
        with_gui: bool = False
    ):
        """Run optimized enhanced backtesting."""
        return self._run_optimized_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="backtesting",
            lookback_days=730,
            with_gui=with_gui,
        )

    def check_optimization_status(self) -> Dict[str, Any]:
        """Check the status of optimization features."""
        return {
            "optimization_enabled": self.optimization_enabled,
            "memory_profiler_active": self.memory_profiler is not None,
            "leak_detector_available": self.leak_detector is not None,
            "optimization_factory_ready": self.optimization_factory is not None
        }


def create_optimized_launcher_patch():
    """Create a patch that can be applied to the existing AresLauncher."""
    
    def patch_launcher(launcher_instance):
        """Apply optimization patches to an existing launcher instance."""
        
        # Add optimization attributes
        launcher_instance.optimization_enabled = True
        launcher_instance.memory_profiler = None
        launcher_instance.leak_detector = None
        launcher_instance.optimization_factory = None
        
        # Add optimization methods
        launcher_instance._setup_optimization_components = OptimizedAresLauncherMixin._setup_optimization_components.__get__(launcher_instance)
        launcher_instance._run_optimized_unified_training = OptimizedAresLauncherMixin._run_optimized_unified_training.__get__(launcher_instance)
        launcher_instance.run_optimized_enhanced_blank_training = OptimizedAresLauncherMixin.run_optimized_enhanced_blank_training.__get__(launcher_instance)
        launcher_instance.run_optimized_backtesting = OptimizedAresLauncherMixin.run_optimized_backtesting.__get__(launcher_instance)
        launcher_instance.check_optimization_status = OptimizedAresLauncherMixin.check_optimization_status.__get__(launcher_instance)
        
        launcher_instance.logger.info("✅ Optimization patches applied to launcher")
        
        return launcher_instance
    
    return patch_launcher


# Quick integration function for immediate use
def enable_optimizations_in_launcher():
    """
    Quick function to enable optimizations in the current launcher.
    This can be called from ares_launcher.py to enable the new features.
    """
    import sys
    from pathlib import Path
    
    # Add the project root to the path if not already there
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import and patch the launcher
    try:
        # This would be used in the actual launcher file
        patch_function = create_optimized_launcher_patch()
        return patch_function
    except Exception as e:
        print(f"❌ Failed to enable optimizations: {e}")
        return None