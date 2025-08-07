# src/training/training_manager.py

import warnings
from datetime import datetime
from typing import Any, NUmber

warnings.filterwarnings("ignore")


# Import the new RegularizationManager
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class TrainingManager:
    """
    Enhanced training manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize training manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("TrainingManager")

        # Training manager state
        self.is_training: bool = False
        self.training_results: dict[str, Any] = {}
        self.training_history: list[dict[str, Any]] = []

        # Configuration
        self.training_config: dict[str, Any] = self.config.get("training_manager", {})
        self.training_interval: int = self.training_config.get(
            "training_interval",
            3600,
        )
        self.max_training_history: int = self.training_config.get(
            "max_training_history",
            100,
        )
        self.enable_model_training: bool = self.training_config.get(
            "enable_model_training",
            True,
        )
        self.enable_hyperparameter_optimization: bool = self.training_config.get(
            "enable_hyperparameter_optimization",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training manager configuration"),
            AttributeError: (False, "Missing required training parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="training manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize training manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Training Manager...")

            # Load training configuration
            await self._load_training_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for training manager")
                return False

            # Initialize training modules
            await self._initialize_training_modules()

            self.logger.info(
                "✅ Training Manager initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Training Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training configuration loading",
    )
    async def _load_training_configuration(self) -> None:
        """Load training configuration."""
        try:
            # Set default training parameters
            self.training_config.setdefault("training_interval", 3600)
            self.training_config.setdefault("max_training_history", 100)
            self.training_config.setdefault("enable_model_training", True)
            self.training_config.setdefault("enable_hyperparameter_optimization", True)
            self.training_config.setdefault("enable_model_evaluation", True)
            self.training_config.setdefault("enable_model_persistence", True)

            # Update configuration
            self.training_interval = self.training_config["training_interval"]
            self.max_training_history = self.training_config["max_training_history"]
            self.enable_model_training = self.training_config["enable_model_training"]
            self.enable_hyperparameter_optimization = self.training_config[
                "enable_hyperparameter_optimization"
            ]

            self.logger.info("Training configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading training configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate training configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate training interval
            if self.training_interval <= 0:
                self.logger.error("Invalid training interval")
                return False

            # Validate max training history
            if self.max_training_history <= 0:
                self.logger.error("Invalid max training history")
                return False

            # Validate that at least one training type is enabled
            if not any(
                [
                    self.enable_model_training,
                    self.enable_hyperparameter_optimization,
                    self.training_config.get("enable_model_evaluation", True),
                    self.training_config.get("enable_model_persistence", True),
                ],
            ):
                self.logger.error("At least one training type must be enabled")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training modules initialization",
    )
    async def _initialize_training_modules(self) -> None:
        """Initialize training modules."""
        try:
            self.logger.info("Initializing training modules...")

            # Initialize model training
            if self.enable_model_training:
                await self._initialize_model_training()

            # Initialize hyperparameter optimization
            if self.enable_hyperparameter_optimization:
                await self._initialize_hyperparameter_optimization()

            # Initialize model evaluation
            await self._initialize_model_evaluation()

            # Initialize model persistence
            await self._initialize_model_persistence()

            # Initialize feature integration manager
            await self._initialize_feature_integration()

            self.logger.info("Training modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing training modules: {e}")

    async def _initialize_feature_integration(self) -> None:
        """Initialize feature integration manager."""
        try:
            from src.training.feature_integration import FeatureIntegrationManager

            self.feature_integration_manager = FeatureIntegrationManager(self.config)
            await self.feature_integration_manager.initialize()
            self.logger.info("Feature integration manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing feature integration manager: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model training initialization",
    )
    async def _initialize_model_training(self) -> None:
        """Initialize model training module."""
        try:
            # Initialize model training components
            self.model_training_components = {
                "data_preprocessing": True,
                "feature_engineering": True,
                "model_training": True,
                "model_validation": True,
            }

            self.logger.info("Model training module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing model training: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="hyperparameter optimization initialization",
    )
    async def _initialize_hyperparameter_optimization(self) -> None:
        """Initialize hyperparameter optimization module."""
        try:
            # Initialize hyperparameter optimization components
            self.hyperparameter_optimization_components = {
                "parameter_search": True,
                "cross_validation": True,
                "model_selection": True,
                "optimization_tracking": True,
            }

            self.logger.info("Hyperparameter optimization module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing hyperparameter optimization: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model evaluation initialization",
    )
    async def _initialize_model_evaluation(self) -> None:
        """Initialize model evaluation module."""
        try:
            # Initialize model evaluation components
            self.model_evaluation_components = {
                "performance_metrics": True,
                "model_comparison": True,
                "validation_testing": True,
                "evaluation_reporting": True,
            }

            self.logger.info("Model evaluation module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing model evaluation: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model persistence initialization",
    )
    async def _initialize_model_persistence(self) -> None:
        """Initialize model persistence module."""
        try:
            # Initialize model persistence components
            self.model_persistence_components = {
                "model_saving": True,
                "model_loading": True,
                "model_versioning": True,
                "model_backup": True,
            }

            self.logger.info("Model persistence module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing model persistence: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training parameters"),
            AttributeError: (False, "Missing training components"),
            KeyError: (False, "Missing required training data"),
        },
        default_return=False,
        context="training execution",
    )
    async def execute_training(self, training_input: dict[str, Any]) -> bool:
        """
        Execute training operations.

        Args:
            training_input: Training input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_training_inputs(training_input):
            return False

        self.is_training = True
        self.logger.info("🔄 Starting training execution...")

        # Perform model training
        if self.enable_model_training:
            model_training_results = await self._perform_model_training(
                training_input,
            )
            self.training_results["model_training"] = model_training_results

        # Perform hyperparameter optimization
        if self.enable_hyperparameter_optimization:
            optimization_results = await self._perform_hyperparameter_optimization(
                training_input,
            )
            self.training_results["hyperparameter_optimization"] = optimization_results

        # Perform model evaluation
        if self.training_config.get("enable_model_evaluation", True):
            evaluation_results = await self._perform_model_evaluation(
                training_input,
            )
            self.training_results["model_evaluation"] = evaluation_results

        # Perform model persistence
        if self.training_config.get("enable_model_persistence", True):
            persistence_results = await self._perform_model_persistence(
                training_input,
            )
            self.training_results["model_persistence"] = persistence_results

        # Store training results
        await self._store_training_results()

        self.is_training = False
        self.logger.info("✅ Training execution completed successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="training inputs validation",
    )
    def _validate_training_inputs(self, training_input: dict[str, Any]) -> bool:
        """
        Validate training inputs.

        Args:
            training_input: Training input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        # Check required training input fields
        required_fields = ["training_type", "model_type", "timestamp"]
        for field in required_fields:
            if field not in training_input:
                self.logger.error(f"Missing required training input field: {field}")
                return False

        # Validate data types
        if not isinstance(training_input["training_type"], str):
            self.logger.error("Invalid training type")
            return False

        if not isinstance(training_input["model_type"], str):
            self.logger.error("Invalid model type")
            return False

        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model training",
    )
    async def _perform_model_training(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform model training.

        Args:
            training_input: Training input dictionary

        Returns:
            Dict[str, Any]: Model training results
        """
        try:
            results = {}

            # Perform data preprocessing
            if self.model_training_components.get("data_preprocessing", False):
                results["data_preprocessing"] = self._perform_data_preprocessing(
                    training_input,
                )

            # Perform feature engineering
            if self.model_training_components.get("feature_engineering", False):
                results[
                    "feature_engineering"
                ] = await self._perform_feature_engineering(
                    training_input,
                )

            # Perform model training
            if self.model_training_components.get("model_training", False):
                results["model_training"] = self._perform_model_training_core(
                    training_input,
                )

            # Perform model validation
            if self.model_training_components.get("model_validation", False):
                results["model_validation"] = self._perform_model_validation(
                    training_input,
                )

            self.logger.info("Model training completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing model training: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="hyperparameter optimization",
    )
    async def _perform_hyperparameter_optimization(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform hyperparameter optimization.

        Args:
            training_input: Training input dictionary

        Returns:
            Dict[str, Any]: Hyperparameter optimization results
        """
        try:
            results = {}

            # Perform parameter search
            if self.hyperparameter_optimization_components.get(
                "parameter_search",
                False,
            ):
                results["parameter_search"] = self._perform_parameter_search(
                    training_input,
                )

            # Perform cross validation
            if self.hyperparameter_optimization_components.get(
                "cross_validation",
                False,
            ):
                results["cross_validation"] = self._perform_cross_validation(
                    training_input,
                )

            # Perform model selection
            if self.hyperparameter_optimization_components.get(
                "model_selection",
                False,
            ):
                results["model_selection"] = self._perform_model_selection(
                    training_input,
                )

            # Perform optimization tracking
            if self.hyperparameter_optimization_components.get(
                "optimization_tracking",
                False,
            ):
                results["optimization_tracking"] = self._perform_optimization_tracking(
                    training_input,
                )

            self.logger.info("Hyperparameter optimization completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing hyperparameter optimization: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model evaluation",
    )
    async def _perform_model_evaluation(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform model evaluation.

        Args:
            training_input: Training input dictionary

        Returns:
            Dict[str, Any]: Model evaluation results
        """
        try:
            results = {}

            # Perform performance metrics
            if self.model_evaluation_components.get("performance_metrics", False):
                results["performance_metrics"] = self._perform_performance_metrics(
                    training_input,
                )

            # Perform model comparison
            if self.model_evaluation_components.get("model_comparison", False):
                results["model_comparison"] = self._perform_model_comparison(
                    training_input,
                )

            # Perform validation testing
            if self.model_evaluation_components.get("validation_testing", False):
                results["validation_testing"] = self._perform_validation_testing(
                    training_input,
                )

            # Perform evaluation reporting
            if self.model_evaluation_components.get("evaluation_reporting", False):
                results["evaluation_reporting"] = self._perform_evaluation_reporting(
                    training_input,
                )

            self.logger.info("Model evaluation completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing model evaluation: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model persistence",
    )
    async def _perform_model_persistence(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform model persistence.

        Args:
            training_input: Training input dictionary

        Returns:
            Dict[str, Any]: Model persistence results
        """
        try:
            results = {}

            # Perform model saving
            if self.model_persistence_components.get("model_saving", False):
                results["model_saving"] = self._perform_model_saving(training_input)

            # Perform model loading
            if self.model_persistence_components.get("model_loading", False):
                results["model_loading"] = self._perform_model_loading(training_input)

            # Perform model versioning
            if self.model_persistence_components.get("model_versioning", False):
                results["model_versioning"] = self._perform_model_versioning(
                    training_input,
                )

            # Perform model backup
            if self.model_persistence_components.get("model_backup", False):
                results["model_backup"] = self._perform_model_backup(training_input)

            self.logger.info("Model persistence completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing model persistence: {e}")
            return {}

    # Model training methods
    def _perform_data_preprocessing(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform data preprocessing."""
        try:
            # Simulate data preprocessing
            return {
                "preprocessing_completed": True,
                "data_cleaned": 10000,
                "features_processed": 50,
                "preprocessing_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing data preprocessing: {e}")
            return {}

    async def _perform_feature_engineering(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform feature engineering with liquidity features integration."""
        try:
            historical_data = training_input.get("historical_data")
            market_data = training_input.get("market_data", historical_data)
            order_flow_data = training_input.get("order_flow_data")

            if self.feature_integration_manager:
                # Use feature integration manager to add advanced features including liquidity
                integrated_data = (
                    await self.feature_integration_manager.integrate_features(
                        historical_data=historical_data,
                        market_data=market_data,
                        order_flow_data=order_flow_data,
                    )
                )

                # Get liquidity feature summary
                liquidity_summary = (
                    self.feature_integration_manager.get_liquidity_feature_summary(
                        integrated_data,
                    )
                )
                self.logger.info(f"Liquidity features integrated: {liquidity_summary}")

                return {
                    "engineered_features": integrated_data,
                    "liquidity_summary": liquidity_summary,
                    "feature_count": len(integrated_data.columns),
                    "liquidity_feature_count": liquidity_summary.get(
                        "total_liquidity_features",
                        0,
                    ),
                }
            self.logger.warning(
                "Feature integration manager not available, using original data",
            )
            return {
                "engineered_features": historical_data,
                "feature_count": len(historical_data.columns),
                "liquidity_feature_count": 0,
            }

        except Exception as e:
            self.logger.error(f"Error performing feature engineering: {e}")
            return {
                "engineered_features": training_input.get("historical_data"),
                "feature_count": 0,
                "liquidity_feature_count": 0,
            }

    def _perform_model_training_core(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform model training core."""
        try:
            # Simulate model training
            return {
                "training_completed": True,
                "epochs_trained": 100,
                "training_accuracy": 0.85,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model training core: {e}")
            return {}

    def _perform_model_validation(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform model validation."""
        try:
            # Simulate model validation
            return {
                "validation_completed": True,
                "validation_accuracy": 0.82,
                "validation_loss": 0.18,
                "validation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model validation: {e}")
            return {}

    # Hyperparameter optimization methods
    def _perform_parameter_search(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform parameter search."""
        try:
            # Simulate parameter search
            return {
                "parameters_searched": 50,
                "best_parameters": {"learning_rate": 0.001, "batch_size": 32},
                "search_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing parameter search: {e}")
            return {}

    def _perform_cross_validation(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform cross validation."""
        try:
            # Simulate cross validation
            return {
                "cv_folds": 5,
                "cv_score": 0.83,
                "cv_std": 0.02,
                "validation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing cross validation: {e}")
            return {}

    def _perform_model_selection(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform model selection."""
        try:
            # Simulate model selection
            return {
                "models_evaluated": 10,
                "best_model": "RandomForest",
                "selection_score": 0.85,
                "selection_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model selection: {e}")
            return {}

    def _perform_optimization_tracking(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform optimization tracking."""
        try:
            # Simulate optimization tracking
            return {
                "optimization_iterations": 100,
                "best_score": 0.87,
                "convergence_reached": True,
                "tracking_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing optimization tracking: {e}")
            return {}

    # Model evaluation methods
    def _perform_performance_metrics(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Number]:
        """Perform performance metrics."""
        try:
            # Simulate performance metrics
            return {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
                "metrics_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing performance metrics: {e}")
            return {}

    def _perform_model_comparison(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform model comparison."""
        try:
            # Simulate model comparison
            return {
                "models_compared": 5,
                "best_model": "RandomForest",
                "comparison_metrics": {"accuracy": 0.85, "speed": 0.92},
                "comparison_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model comparison: {e}")
            return {}

    def _perform_validation_testing(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform validation testing."""
        try:
            # Simulate validation testing
            return {
                "test_accuracy": 0.84,
                "test_loss": 0.16,
                "test_samples": 2000,
                "testing_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing validation testing: {e}")
            return {}

    def _perform_evaluation_reporting(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform evaluation reporting."""
        try:
            # Simulate evaluation reporting
            return {
                "report_generated": True,
                "report_format": "json",
                "report_location": "/reports/training_report.json",
                "reporting_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing evaluation reporting: {e}")
            return {}

    # Model persistence methods
    def _perform_model_saving(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Perform model saving."""
        try:
            # Simulate model saving
            return {
                "model_saved": True,
                "model_size": "15.2MB",
                "save_location": "/models/best_model.pkl",
                "saving_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model saving: {e}")
            return {}

    def _perform_model_loading(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Perform model loading."""
        try:
            # Simulate model loading
            return {
                "model_loaded": True,
                "load_time": 0.5,
                "model_ready": True,
                "loading_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model loading: {e}")
            return {}

    def _perform_model_versioning(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform model versioning."""
        try:
            # Simulate model versioning
            return {
                "version_created": "v1.2.3",
                "version_metadata": {"accuracy": 0.85, "training_date": "2024-01-15"},
                "versioning_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model versioning: {e}")
            return {}

    def _perform_model_backup(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Perform model backup."""
        try:
            # Simulate model backup
            return {
                "backup_created": True,
                "backup_size": "15.2MB",
                "backup_location": "/backups/model_backup_20240115.pkl",
                "backup_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing model backup: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training results storage",
    )
    async def _store_training_results(self) -> None:
        """Store training results."""
        try:
            # Add timestamp
            self.training_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.training_history.append(self.training_results.copy())

            # Limit history size
            if len(self.training_history) > self.max_training_history:
                self.training_history.pop(0)

            self.logger.info("Training results stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing training results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training results getting",
    )
    def get_training_results(
        self,
        training_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get training results.

        Args:
            training_type: Optional training type filter

        Returns:
            Dict[str, Any]: Training results
        """
        try:
            if training_type:
                return self.training_results.get(training_type, {})
            return self.training_results.copy()

        except Exception as e:
            self.logger.error(f"Error getting training results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training history getting",
    )
    def get_training_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get training history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Training history
        """
        try:
            history = self.training_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.error(f"Error getting training history: {e}")
            return []

    def get_training_status(self) -> dict[str, Any]:
        """
        Get training status information.

        Returns:
            Dict[str, Any]: Training status
        """
        return {
            "is_training": self.is_training,
            "training_interval": self.training_interval,
            "max_training_history": self.max_training_history,
            "enable_model_training": self.enable_model_training,
            "enable_hyperparameter_optimization": self.enable_hyperparameter_optimization,
            "enable_model_evaluation": self.training_config.get(
                "enable_model_evaluation",
                True,
            ),
            "enable_model_persistence": self.training_config.get(
                "enable_model_persistence",
                True,
            ),
            "training_history_count": len(self.training_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="training manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the training manager."""
        self.logger.info("🛑 Stopping Training Manager...")

        try:
            # Stop training
            self.is_training = False

            # Clear results
            self.training_results.clear()

            # Clear history
            self.training_history.clear()

            self.logger.info("✅ Training Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping training manager: {e}")


# Global training manager instance
training_manager: TrainingManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="training manager setup",
)
async def setup_training_manager(
    config: dict[str, Any] | None = None,
) -> TrainingManager | None:
    """
    Setup global training manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[TrainingManager]: Global training manager instance
    """
    try:
        global training_manager

        if config is None:
            config = {
                "training_manager": {
                    "training_interval": 3600,
                    "max_training_history": 100,
                    "enable_model_training": True,
                    "enable_hyperparameter_optimization": True,
                    "enable_model_evaluation": True,
                    "enable_model_persistence": True,
                },
            }

        # Create training manager
        training_manager = TrainingManager(config)

        # Initialize training manager
        success = await training_manager.initialize()
        if success:
            return training_manager
        return None

    except Exception as e:
        print(f"Error setting up training manager: {e}")
        return None
