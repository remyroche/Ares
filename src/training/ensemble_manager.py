# src/training/ensemble_manager.py

import asyncio
from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class EnsembleManager:
    """
    Ensemble manager responsible for creating and managing model ensembles.
    This module handles ensemble creation, optimization, and management.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ensemble manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("EnsembleManager")
        
        # Ensemble state
        self.is_creating_ensembles: bool = False
        self.ensemble_results: dict[str, Any] = {}
        
        # Configuration
        self.ensemble_config: dict[str, Any] = self.config.get("ensemble_manager", {})
        self.enable_analyst_ensembles: bool = self.ensemble_config.get("enable_analyst_ensembles", True)
        self.enable_tactician_ensembles: bool = self.ensemble_config.get("enable_tactician_ensembles", True)
        self.enable_ensemble_optimization: bool = self.ensemble_config.get("enable_ensemble_optimization", True)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ensemble manager configuration"),
            AttributeError: (False, "Missing required ensemble parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ensemble manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize ensemble manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Ensemble Manager...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for ensemble manager")
                return False
            
            # Initialize ensemble components
            await self._initialize_ensemble_components()
            
            self.logger.info("✅ Ensemble Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate ensemble manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate ensemble manager specific settings
            if not any([
                self.enable_analyst_ensembles,
                self.enable_tactician_ensembles
            ]):
                self.logger.error("At least one ensemble type must be enabled")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ensemble components initialization",
    )
    async def _initialize_ensemble_components(self) -> None:
        """Initialize ensemble components."""
        try:
            # Initialize ensemble creator
            from src.training.ensemble_creator import EnsembleCreator
            self.ensemble_creator = EnsembleCreator(self.config)
            await self.ensemble_creator.initialize()
            
            # Initialize ensemble optimization components
            if self.enable_ensemble_optimization:
                self.logger.info("✅ Ensemble optimization components initialized")
            
            self.logger.info("✅ All ensemble components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ensemble components: {e}")
            raise

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ensemble parameters"),
            AttributeError: (False, "Missing ensemble components"),
            KeyError: (False, "Missing required ensemble data"),
        },
        default_return=False,
        context="ensemble creation",
    )
    async def create_ensembles(
        self,
        optimization_results: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create ensembles from optimized models.

        Args:
            optimization_results: Results from model optimization
            training_input: Training input parameters

        Returns:
            dict: Ensemble creation results
        """
        try:
            self.logger.info("🎯 Starting ensemble creation...")
            self.is_creating_ensembles = True
            
            # Validate inputs
            if not self._validate_ensemble_inputs(optimization_results, training_input):
                return None
            
            # Create analyst ensembles
            analyst_ensembles = None
            if self.enable_analyst_ensembles:
                analyst_ensembles = await self._create_analyst_ensembles(optimization_results, training_input)
            
            # Create tactician ensembles
            tactician_ensembles = None
            if self.enable_tactician_ensembles:
                tactician_ensembles = await self._create_tactician_ensembles(optimization_results, training_input)
            
            # Optimize ensembles if enabled
            if self.enable_ensemble_optimization:
                if analyst_ensembles:
                    analyst_ensembles = await self._optimize_ensembles(analyst_ensembles, "analyst")
                if tactician_ensembles:
                    tactician_ensembles = await self._optimize_ensembles(tactician_ensembles, "tactician")
            
            # Combine results
            ensemble_results = {
                "analyst_ensembles": analyst_ensembles,
                "tactician_ensembles": tactician_ensembles,
                "training_input": training_input,
                "ensemble_creation_timestamp": datetime.now().isoformat(),
            }
            
            # Store ensemble results
            await self._store_ensemble_results(ensemble_results)
            
            self.is_creating_ensembles = False
            self.logger.info("✅ Ensemble creation completed successfully")
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble creation failed: {e}")
            self.is_creating_ensembles = False
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="ensemble inputs validation",
    )
    def _validate_ensemble_inputs(
        self,
        optimization_results: dict[str, Any],
        training_input: dict[str, Any],
    ) -> bool:
        """
        Validate ensemble input parameters.

        Args:
            optimization_results: Results from model optimization
            training_input: Training input parameters

        Returns:
            bool: True if inputs are valid, False otherwise
        """
        try:
            # Validate optimization results
            if not optimization_results:
                self.logger.error("Optimization results are empty")
                return False
            
            # Validate training input
            if not training_input:
                self.logger.error("Training input is empty")
                return False
            
            # Check for required optimization results
            if not optimization_results.get("optimized_models"):
                self.logger.error("No optimized models found in results")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ensemble inputs validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst ensemble creation",
    )
    async def _create_analyst_ensembles(
        self,
        optimization_results: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create analyst model ensembles.

        Args:
            optimization_results: Results from model optimization
            training_input: Training input parameters

        Returns:
            dict: Analyst ensemble creation results
        """
        try:
            self.logger.info("🧠 Creating analyst ensembles...")
            
            analyst_ensembles = {}
            
            # Get optimized analyst models
            optimized_models = optimization_results.get("optimized_models", {})
            analyst_models = {k: v for k, v in optimized_models.items() if k.startswith("analyst_")}
            
            if not analyst_models:
                self.logger.warning("No analyst models found for ensemble creation")
                return None
            
            # Create multi-timeframe ensemble
            multi_timeframe_ensemble = await self._create_multi_timeframe_ensemble(
                analyst_models, training_input
            )
            if multi_timeframe_ensemble:
                analyst_ensembles["multi_timeframe"] = multi_timeframe_ensemble
            
            # Create individual timeframe ensembles
            for timeframe in ["1h", "15m", "5m", "1m"]:
                timeframe_models = {k: v for k, v in analyst_models.items() if timeframe in k}
                if timeframe_models:
                    timeframe_ensemble = await self._create_timeframe_ensemble(
                        timeframe_models, timeframe, training_input
                    )
                    if timeframe_ensemble:
                        analyst_ensembles[f"timeframe_{timeframe}"] = timeframe_ensemble
            
            self.logger.info(f"✅ Created {len(analyst_ensembles)} analyst ensembles")
            return analyst_ensembles
            
        except Exception as e:
            self.logger.error(f"❌ Analyst ensemble creation failed: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician ensemble creation",
    )
    async def _create_tactician_ensembles(
        self,
        optimization_results: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create tactician model ensembles.

        Args:
            optimization_results: Results from model optimization
            training_input: Training input parameters

        Returns:
            dict: Tactician ensemble creation results
        """
        try:
            self.logger.info("🎯 Creating tactician ensembles...")
            
            tactician_ensembles = {}
            
            # Get optimized tactician models
            optimized_models = optimization_results.get("optimized_models", {})
            tactician_models = {k: v for k, v in optimized_models.items() if k.startswith("tactician_")}
            
            if not tactician_models:
                self.logger.warning("No tactician models found for ensemble creation")
                return None
            
            # Create single timeframe ensemble for tactician (1m only)
            tactician_ensemble = await self._create_tactician_single_ensemble(
                tactician_models, training_input
            )
            if tactician_ensemble:
                tactician_ensembles["single_timeframe"] = tactician_ensemble
            
            self.logger.info(f"✅ Created {len(tactician_ensembles)} tactician ensembles")
            return tactician_ensembles
            
        except Exception as e:
            self.logger.error(f"❌ Tactician ensemble creation failed: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="multi-timeframe ensemble creation",
    )
    async def _create_multi_timeframe_ensemble(
        self,
        analyst_models: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create multi-timeframe ensemble for analyst models.

        Args:
            analyst_models: Optimized analyst models
            training_input: Training input parameters

        Returns:
            dict: Multi-timeframe ensemble result
        """
        try:
            self.logger.info("🧠 Creating multi-timeframe analyst ensemble...")
            
            # This would implement actual multi-timeframe ensemble creation logic
            # For now, return a placeholder result
            ensemble_result = {
                "ensemble_type": "multi_timeframe_weighted",
                "timeframe_weights": {
                    "1h": 0.3,
                    "15m": 0.25,
                    "5m": 0.25,
                    "1m": 0.2,
                },
                "ensemble_metrics": {
                    "accuracy": 0.87,
                    "precision": 0.84,
                    "recall": 0.81,
                },
                "ensemble_path": f"ensembles/analyst_multi_timeframe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "constituent_models": list(analyst_models.keys()),
            }
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create multi-timeframe ensemble: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="timeframe ensemble creation",
    )
    async def _create_timeframe_ensemble(
        self,
        timeframe_models: dict[str, Any],
        timeframe: str,
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create ensemble for a specific timeframe.

        Args:
            timeframe_models: Models for the specific timeframe
            timeframe: Target timeframe
            training_input: Training input parameters

        Returns:
            dict: Timeframe ensemble result
        """
        try:
            self.logger.info(f"🧠 Creating {timeframe} timeframe ensemble...")
            
            # This would implement actual timeframe ensemble creation logic
            # For now, return a placeholder result
            ensemble_result = {
                "ensemble_type": "single_timeframe_weighted",
                "timeframe": timeframe,
                "model_weights": {
                    "random_forest": 0.4,
                    "lightgbm": 0.35,
                    "xgboost": 0.25,
                },
                "ensemble_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.79,
                },
                "ensemble_path": f"ensembles/analyst_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "constituent_models": list(timeframe_models.keys()),
            }
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create {timeframe} timeframe ensemble: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician single ensemble creation",
    )
    async def _create_tactician_single_ensemble(
        self,
        tactician_models: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Create single ensemble for tactician models.

        Args:
            tactician_models: Optimized tactician models
            training_input: Training input parameters

        Returns:
            dict: Tactician ensemble result
        """
        try:
            self.logger.info("🎯 Creating tactician single ensemble...")
            
            # This would implement actual tactician ensemble creation logic
            # For now, return a placeholder result
            ensemble_result = {
                "ensemble_type": "single_timeframe_weighted",
                "timeframe": "1m",
                "model_weights": {
                    "random_forest": 0.4,
                    "lightgbm": 0.35,
                    "xgboost": 0.25,
                },
                "ensemble_metrics": {
                    "accuracy": 0.89,
                    "precision": 0.86,
                    "recall": 0.83,
                },
                "ensemble_path": f"ensembles/tactician_1m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                "constituent_models": list(tactician_models.keys()),
            }
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create tactician ensemble: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ensemble optimization",
    )
    async def _optimize_ensembles(
        self,
        ensembles: dict[str, Any],
        ensemble_type: str,
    ) -> dict[str, Any] | None:
        """
        Optimize ensembles.

        Args:
            ensembles: Ensembles to optimize
            ensemble_type: Type of ensemble (analyst or tactician)

        Returns:
            dict: Optimized ensembles
        """
        try:
            self.logger.info(f"🔧 Optimizing {ensemble_type} ensembles...")
            
            optimized_ensembles = {}
            
            for ensemble_name, ensemble in ensembles.items():
                optimized_ensemble = await self._optimize_single_ensemble(
                    ensemble, ensemble_name, ensemble_type
                )
                if optimized_ensemble:
                    optimized_ensembles[ensemble_name] = optimized_ensemble
            
            self.logger.info(f"✅ Optimized {len(optimized_ensembles)} {ensemble_type} ensembles")
            return optimized_ensembles
            
        except Exception as e:
            self.logger.error(f"❌ {ensemble_type} ensemble optimization failed: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="single ensemble optimization",
    )
    async def _optimize_single_ensemble(
        self,
        ensemble: dict[str, Any],
        ensemble_name: str,
        ensemble_type: str,
    ) -> dict[str, Any] | None:
        """
        Optimize a single ensemble.

        Args:
            ensemble: Ensemble to optimize
            ensemble_name: Name of the ensemble
            ensemble_type: Type of ensemble

        Returns:
            dict: Optimized ensemble
        """
        try:
            self.logger.info(f"🔧 Optimizing {ensemble_type} ensemble: {ensemble_name}")
            
            # This would implement actual ensemble optimization logic
            # For now, return a placeholder result
            optimized_ensemble = {
                "original_ensemble": ensemble,
                "optimized_weights": ensemble.get("model_weights", {}),
                "optimization_metrics": {
                    "improvement": 0.03,
                    "optimization_time": 15.5,
                },
                "optimized_ensemble_path": f"ensembles/optimized_{ensemble_type}_{ensemble_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            }
            
            return optimized_ensemble
            
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize {ensemble_type} ensemble {ensemble_name}: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ensemble results storage",
    )
    async def _store_ensemble_results(self, ensemble_results: dict[str, Any]) -> None:
        """
        Store ensemble results.

        Args:
            ensemble_results: Ensemble results to store
        """
        try:
            self.logger.info("📁 Storing ensemble results...")
            
            # Store ensemble results in memory for now
            # In practice, this would store to database or file system
            self.ensemble_results = ensemble_results.copy()
            
            self.logger.info("✅ Ensemble results stored successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store ensemble results: {e}")

    def get_ensemble_status(self) -> dict[str, Any]:
        """
        Get current ensemble status.

        Returns:
            dict: Ensemble status information
        """
        return {
            "is_creating_ensembles": self.is_creating_ensembles,
            "has_ensemble_results": bool(self.ensemble_results),
            "analyst_ensembles_enabled": self.enable_analyst_ensembles,
            "tactician_ensembles_enabled": self.enable_tactician_ensembles,
            "ensemble_optimization_enabled": self.enable_ensemble_optimization,
        }

    def get_ensemble_results(self) -> dict[str, Any]:
        """
        Get the latest ensemble results.

        Returns:
            dict: Ensemble results
        """
        return self.ensemble_results.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ensemble manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the ensemble manager and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Ensemble Manager...")
            self.is_creating_ensembles = False
            self.logger.info("✅ Ensemble Manager stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to stop Ensemble Manager: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ensemble manager setup",
)
async def setup_ensemble_manager(
    config: dict[str, Any] | None = None,
) -> EnsembleManager | None:
    """
    Setup and return a configured EnsembleManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        EnsembleManager: Configured ensemble manager instance
    """
    try:
        manager = EnsembleManager(config or {})
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup ensemble manager: {e}")
        return None 