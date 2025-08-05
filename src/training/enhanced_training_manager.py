# src/training/enhanced_training_manager.py

import asyncio
from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


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
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid configuration for enhanced training manager")
                return False
                
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
            
            # Step 1: Data Collection
            self.logger.info("📊 STEP 1: Data Collection...")
            print("   📊 Step 1: Data Collection...")
            
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
            
            self.logger.info("✅ Step 1: Data Collection completed successfully")
            print("   ✅ Step 1: Data Collection completed successfully")

            # Step 2: Market Regime Classification
            self.logger.info("🎭 STEP 2: Market Regime Classification...")
            print("   🎭 Step 2: Market Regime Classification...")
            
            from src.training.steps import step2_market_regime_classification
            step2_success = await step2_market_regime_classification.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step2_success:
                self.logger.error("❌ Step 2: Market Regime Classification failed")
                print("❌ Step 2: Market Regime Classification failed")
                return False
            
            self.logger.info("✅ Step 2: Market Regime Classification completed successfully")
            print("   ✅ Step 2: Market Regime Classification completed successfully")

            # Step 3: Regime Data Splitting
            self.logger.info("📊 STEP 3: Regime Data Splitting...")
            print("   📊 Step 3: Regime Data Splitting...")
            
            from src.training.steps import step3_regime_data_splitting
            step3_success = await step3_regime_data_splitting.run_step(
                symbol=symbol,
                data_dir=data_dir,
                timeframe=timeframe,
                exchange=exchange,
            )
            
            if not step3_success:
                self.logger.error("❌ Step 3: Regime Data Splitting failed")
                print("❌ Step 3: Regime Data Splitting failed")
                return False
            
            self.logger.info("✅ Step 3: Regime Data Splitting completed successfully")
            print("   ✅ Step 3: Regime Data Splitting completed successfully")

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
                self.logger.error("❌ Step 4: Analyst Labeling & Feature Engineering failed")
                print("❌ Step 4: Analyst Labeling & Feature Engineering failed")
                return False
            
            self.logger.info("✅ Step 4: Analyst Labeling & Feature Engineering completed successfully")
            print("   ✅ Step 4: Analyst Labeling & Feature Engineering completed successfully")

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
            
            if not step5_success:
                self.logger.error("❌ Step 5: Analyst Specialist Training failed")
                print("❌ Step 5: Analyst Specialist Training failed")
                return False
            
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

            # Step 12: Final Parameters Optimization
            self.logger.info("🔧 STEP 12: Final Parameters Optimization...")
            print("   🔧 Step 12: Final Parameters Optimization...")
            
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

            return True
            
        except Exception as e:
            self.logger.error(f"💥 COMPREHENSIVE PIPELINE FAILED: {str(e)}")
            self.logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            print(f"💥 COMPREHENSIVE PIPELINE FAILED: {str(e)}")
            print(f"📋 Error details: {type(e).__name__}: {str(e)}")
            return False

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
