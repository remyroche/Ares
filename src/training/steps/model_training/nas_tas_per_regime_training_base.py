"""Shared utilities for per-regime NAS/TAS training steps."""

from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

from src.utils.logger import system_logger


@dataclass
class BasePerRegimeTrainingConfig:
    """Common configuration flags shared across NAS and TAS training steps."""

    n_regimes: int = 8
    primary_timeframe: str = ""
    enable_hpo: bool = True
    enable_cv: bool = True
    enable_walk_forward: bool = True
    enable_lookahead_prevention: bool = True


class PerRegimeTrainingStep(ABC):
    """Template for NAS/TAS style training steps that operate per regime."""

    def __init__(
        self,
        *,
        config: BasePerRegimeTrainingConfig,
        logger_name: str,
        step_name: str,
        model_prefix: str,
        display_name: str,
    ) -> None:
        self.config = config
        self.logger = system_logger.getChild(logger_name)
        self.step_name = step_name
        self.model_prefix = model_prefix
        self.display_name = display_name

        # Mutable state shared across subclasses.
        self.models: Dict[Any, Any] = {}
        self.architectures: Dict[Any, Any] = {}
        self.hyperparameters: Dict[Any, Any] = {}
        self.training_history = []
        self.performance_metrics: Dict[str, Any] = {}

        self.logger.info(
            "✅ %s Training Step initialized", self.display_name.upper()
        )
        self.logger.info("   Timeframe: %s", config.primary_timeframe)
        self.logger.info("   Regimes: %s", config.n_regimes)
        self.logger.info("   HPO enabled: %s", config.enable_hpo)
        self.logger.info("   CV enabled: %s", config.enable_cv)

    async def execute_training(
        self, training_input: Mapping[str, Any], pipeline_state: MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        """Run the shared training pipeline template and delegate specifics to hooks."""

        del pipeline_state  # Subclasses currently operate statelessly.

        start_time = time.time()
        self.logger.info(
            "🚀 Starting %s training for per-regime model discovery...",
            self.display_name.upper(),
        )

        try:
            extracted_inputs = self._extract_training_data(training_input)
        except ValueError as exc:
            execution_time = time.time() - start_time
            self.logger.error("❌ %s training failed: %s", self.display_name.upper(), exc)
            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(exc),
                "step_name": self.step_name,
                "metadata": {"error": str(exc)},
            }

        try:
            architectures = await self._perform_architecture_search(**extracted_inputs)
            hyperparameters = await self._perform_hyperparameter_optimization(
                architectures=architectures, **extracted_inputs
            )
            models = await self._train_models(
                architectures=architectures,
                hyperparameters=hyperparameters,
                **extracted_inputs,
            )
            validation_results = await self._validate_models(
                models=models, **extracted_inputs
            )

            execution_time = time.time() - start_time
            metadata = self._build_metadata(
                extracted_inputs,
                architectures=architectures,
                hyperparameters=hyperparameters,
                models=models,
                validation_results=validation_results,
            )

            results = {
                "success": True,
                "execution_time": execution_time,
                "step_name": self.step_name,
                f"{self.model_prefix}_architectures": architectures,
                f"{self.model_prefix}_hyperparameters": hyperparameters,
                f"{self.model_prefix}_models": models,
                "validation_results": validation_results,
                "metadata": metadata,
            }

            self.logger.info(
                "✅ %s training completed in %.2fs",
                self.display_name.upper(),
                execution_time,
            )
            self._log_training_summary(results)
            return results

        except Exception as exc:  # pragma: no cover - defensive logging
            execution_time = time.time() - start_time
            self.logger.error("❌ %s training failed: %s", self.display_name.upper(), exc)
            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(exc),
                "step_name": self.step_name,
                "metadata": {"error": str(exc)},
            }

    @abstractmethod
    def _extract_training_data(
        self, training_input: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Validate and extract raw inputs needed for the pipeline."""

    @abstractmethod
    async def _perform_architecture_search(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the per-regime architecture search for the subclass."""

    @abstractmethod
    async def _perform_hyperparameter_optimization(
        self, *, architectures: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """Tune hyperparameters for the discovered architectures."""

    @abstractmethod
    async def _train_models(
        self,
        *,
        architectures: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train models per regime for the subclass."""

    @abstractmethod
    async def _validate_models(
        self, *, models: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """Validate trained models."""

    @abstractmethod
    def _build_metadata(
        self,
        extracted_inputs: Mapping[str, Any],
        *,
        architectures: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        models: Dict[str, Any],
        validation_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble metadata for logging and downstream consumption."""

    @abstractmethod
    def _sync_aliases(self) -> None:
        """Allow subclasses to maintain backwards-compatible attribute aliases."""

    def _log_training_summary(self, results: Mapping[str, Any]) -> None:
        """Log a unified summary for NAS/TAS training steps."""

        metadata = results.get("metadata", {})
        self.logger.info("📊 %s Training Summary:", self.display_name.upper())
        self.logger.info("   Success: %s", results.get("success", False))
        self.logger.info(
            "   Execution time: %.2fs", results.get("execution_time", 0.0)
        )
        self.logger.info(
            "   Timeframe: %s", metadata.get("timeframe", "unknown")
        )
        self.logger.info("   Regimes: %s", metadata.get("n_regimes", 0))
        self.logger.info(
            "   %s models trained: %s",
            self.display_name.upper(),
            metadata.get(f"{self.model_prefix}_models_trained", 0),
        )
        self.logger.info("   HPO enabled: %s", metadata.get("hpo_enabled", False))
        self.logger.info("   CV enabled: %s", metadata.get("cv_enabled", False))
        self.logger.info(
            "   Walk forward enabled: %s",
            metadata.get("walk_forward_enabled", False),
        )
        self.logger.info(
            "   Lookahead prevention enabled: %s",
            metadata.get("lookahead_prevention_enabled", False),
        )

    def save_models(self, filepath: str) -> bool:
        """Persist trained model artefacts to disk."""

        try:
            model_data = {
                f"{self.model_prefix}_models": self.models,
                f"{self.model_prefix}_architectures": self.architectures,
                f"{self.model_prefix}_hyperparameters": self.hyperparameters,
                "config": self.config,
                "training_history": self.training_history,
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as handle:
                pickle.dump(model_data, handle)

            self.logger.info(
                "✅ %s models saved to %s", self.display_name.upper(), filepath
            )
            return True

        except Exception as exc:  # pragma: no cover - persistence failures are logged
            self.logger.error(
                "❌ Failed to save %s models: %s", self.display_name.upper(), exc
            )
            return False

    def load_models(self, filepath: str) -> bool:
        """Load trained model artefacts from disk."""

        try:
            with open(filepath, "rb") as handle:
                model_data = pickle.load(handle)

            self.models = model_data.get(f"{self.model_prefix}_models", {})
            self.architectures = model_data.get(
                f"{self.model_prefix}_architectures", {}
            )
            self.hyperparameters = model_data.get(
                f"{self.model_prefix}_hyperparameters", {}
            )
            self.training_history = model_data.get("training_history", [])

            self._sync_aliases()

            self.logger.info(
                "✅ %s models loaded from %s", self.display_name.upper(), filepath
            )
            return True

        except Exception as exc:  # pragma: no cover - persistence failures are logged
            self.logger.error(
                "❌ Failed to load %s models: %s", self.display_name.upper(), exc
            )
            return False

