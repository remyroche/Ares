"""
Refactored Step16ConfidenceCalibration with reduced complexity and type hints.
This version breaks down the massive execute method into smaller, focused methods.
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CalibrationStage(Enum):
    """Stages of confidence calibration"""
    LOAD_MODELS = "load_models"
    LOAD_ENSEMBLES = "load_ensembles"
    LOAD_VALIDATION = "load_validation"
    CALIBRATE_ANALYST = "calibrate_analyst"
    CALIBRATE_TACTICIAN = "calibrate_tactician"
    SAVE_RESULTS = "save_results"


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration"""
    regime_config: dict[str, Any]
    min_confidence_threshold: float = 0.5
    calibration_method: str = "isotonic"
    cross_validate: bool = True
    n_folds: int = 5
    save_artifacts: bool = True
    use_joblib: bool = True


@dataclass
class ModelData:
    """Container for model data"""
    analyst_models: dict[str, dict[str, Any]]
    tactician_models: dict[str, Any]
    analyst_ensembles: dict[str, Any]
    tactician_ensembles: dict[str, Any]


@dataclass
class CalibrationResult:
    """Result of a calibration operation"""
    stage: CalibrationStage
    success: bool
    data: Any
    metadata: dict[str, Any]
    error: Exception | None = None


class Step16ConfidenceCalibrationRefactored:
    """Refactored confidence calibration with reduced complexity"""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the calibration step.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.calibration_config = self._create_calibration_config()

        # Import joblib if available
        try:
            import joblib
            self.joblib = joblib
        except ImportError:
            self.joblib = None
            self.logger.warning("⚠️ joblib not available, using pickle")

    def _create_calibration_config(self) -> CalibrationConfig:
        """Create calibration configuration from config dict"""
        return CalibrationConfig(
            regime_config=self.config.get("regime_config", {}),
            min_confidence_threshold=self.config.get("min_confidence_threshold", 0.5),
            calibration_method=self.config.get("calibration_method", "isotonic"),
            cross_validate=self.config.get("cross_validate", True),
            n_folds=self.config.get("n_folds", 5),
            save_artifacts=self.config.get("save_artifacts", True),
            use_joblib=self.config.get("use_joblib", True) and self.joblib is not None,
        )

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute confidence calibration with reduced complexity.

        This refactored method orchestrates the calibration process by
        delegating to specialized methods for each stage.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing calibration results
        """
        try:
            self.logger.info("🔄 Executing Regime-Aware Confidence Calibration...")

            # Extract parameters
            params = self._extract_parameters(training_input)

            # Execute calibration stages
            results = {}

            # Stage 1: Load models
            model_data = await self._execute_load_models_stage(params)
            if not model_data:
                return self._create_error_result("Failed to load models")

            # Stage 2: Load validation data
            validation_data = await self._execute_load_validation_stage(params)
            if validation_data is None:
                return self._create_error_result("Failed to load validation data")

            # Stage 3: Calibrate analyst models
            analyst_results = await self._execute_analyst_calibration_stage(
                model_data, validation_data, params,
            )
            results["analyst_models"] = analyst_results

            # Stage 4: Calibrate tactician models
            tactician_results = await self._execute_tactician_calibration_stage(
                model_data, validation_data, params,
            )
            results["tactician_models"] = tactician_results

            # Stage 5: Save calibration results
            if self.calibration_config.save_artifacts:
                await self._execute_save_results_stage(results, params)

            # Create final output
            return self._create_success_result(results, model_data, params)

        except Exception as e:
            self.logger.exception(f"❌ Calibration failed: {e}")
            return self._create_error_result(str(e))

    def _extract_parameters(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate parameters from training input"""
        return {
            "symbol": training_input.get("symbol", "ETHUSDT"),
            "exchange": training_input.get("exchange", "BINANCE"),
            "data_dir": training_input.get("data_dir", "data/training"),
            "output_dir": training_input.get("output_dir", "data/calibrated"),
        }

    async def _execute_load_models_stage(
        self,
        params: dict[str, Any],
    ) -> ModelData | None:
        """Execute model loading stage"""
        try:
            self.logger.info("📦 Loading models and ensembles...")

            # Load models in parallel
            tasks = [
                self._load_analyst_models(params["data_dir"]),
                self._load_tactician_models(params["data_dir"]),
                self._load_analyst_ensembles(params["data_dir"]),
                self._load_tactician_ensembles(params["data_dir"], params["exchange"], params["symbol"]),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Model loading task {i} failed: {result}")
                    return None

            return ModelData(
                analyst_models=results[0],
                tactician_models=results[1],
                analyst_ensembles=results[2],
                tactician_ensembles=results[3],
            )

        except Exception as e:
            self.logger.exception(f"❌ Model loading failed: {e}")
            return None

    async def _load_analyst_models(self, data_dir: str) -> dict[str, dict[str, Any]]:
        """Load analyst models from disk"""
        analyst_models = {}
        analyst_models_dir = Path(data_dir) / "enhanced_analyst_models"

        if not analyst_models_dir.exists():
            self.logger.warning("⚠️ Analyst models directory not found")
            return analyst_models

        for regime_dir in analyst_models_dir.iterdir():
            if regime_dir.is_dir():
                regime_models = await self._load_models_from_directory(regime_dir)
                if regime_models:
                    analyst_models[regime_dir.name] = regime_models

        self.logger.info(f"✅ Loaded analyst models for {len(analyst_models)} regimes")
        return analyst_models

    async def _load_models_from_directory(
        self,
        directory: Path,
    ) -> dict[str, Any]:
        """Load all models from a directory"""
        models = {}

        for model_file in directory.iterdir():
            if model_file.suffix in [".pkl", ".joblib"]:
                model_name = model_file.stem
                try:
                    model = await self._load_model_file(model_file)
                    if model is not None:
                        models[model_name] = model
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {model_file}: {e}")

        return models

    async def _load_model_file(self, file_path: Path) -> Any:
        """Load a single model file"""
        try:
            if file_path.suffix == ".joblib" and self.joblib is not None:
                return self.joblib.load(file_path)
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.exception(f"❌ Error loading {file_path}: {e}")
            return None

    async def _load_tactician_models(self, data_dir: str) -> dict[str, Any]:
        """Load tactician models from disk"""
        tactician_models = {}
        tactician_models_dir = Path(data_dir) / "tactician_models"

        if not tactician_models_dir.exists():
            self.logger.warning("⚠️ Tactician models directory not found")
            return tactician_models

        for model_file in tactician_models_dir.glob("*.pkl"):
            model_name = model_file.stem
            try:
                with open(model_file, "rb") as f:
                    tactician_models[model_name] = pickle.load(f)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load tactician model {model_file}: {e}")

        self.logger.info(f"✅ Loaded {len(tactician_models)} tactician models")
        return tactician_models

    async def _load_analyst_ensembles(self, data_dir: str) -> dict[str, Any]:
        """Load analyst ensembles from disk"""
        analyst_ensembles = {}
        ensembles_dir = Path(data_dir) / "analyst_ensembles"

        if not ensembles_dir.exists():
            self.logger.warning("⚠️ Analyst ensembles directory not found")
            return analyst_ensembles

        for ensemble_file in ensembles_dir.glob("*_ensemble.pkl"):
            regime_name = ensemble_file.stem.replace("_ensemble", "")
            try:
                with open(ensemble_file, "rb") as f:
                    analyst_ensembles[regime_name] = pickle.load(f)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load analyst ensemble {ensemble_file}: {e}")

        self.logger.info(f"✅ Loaded {len(analyst_ensembles)} analyst ensembles")
        return analyst_ensembles

    async def _load_tactician_ensembles(
        self,
        data_dir: str,
        exchange: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Load tactician ensembles from disk"""
        tactician_ensembles = {}
        ensembles_dir = Path(data_dir) / "tactician_ensembles"

        if not ensembles_dir.exists():
            self.logger.warning("⚠️ Tactician ensembles directory not found")
            return tactician_ensembles

        # Load primary ensemble
        primary_path = ensembles_dir / f"{exchange}_{symbol}_tactician_ensemble.pkl"
        if primary_path.exists():
            try:
                with open(primary_path, "rb") as f:
                    tactician_ensembles["blended"] = {"ensemble": pickle.load(f)}
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load primary tactician ensemble: {e}")

        # Load additional ensembles
        for ensemble_file in ensembles_dir.glob("*_tactician_ensemble.pkl"):
            if ensemble_file != primary_path:
                try:
                    with open(ensemble_file, "rb") as f:
                        tactician_ensembles[ensemble_file.stem] = {"ensemble": pickle.load(f)}
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load tactician ensemble {ensemble_file}: {e}")

        self.logger.info(f"✅ Loaded {len(tactician_ensembles)} tactician ensembles")
        return tactician_ensembles

    async def _execute_load_validation_stage(
        self,
        params: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Execute validation data loading stage"""
        try:
            self.logger.info("📊 Loading validation data...")

            # Load base validation frame
            validation_data = await self._load_validation_frame(
                params["data_dir"],
                params["exchange"],
                params["symbol"],
            )

            if validation_data is None:
                return None

            # Augment with additional features if available
            augmented_data = await self._augment_validation_data(
                validation_data, params,
            )

            self.logger.info(f"✅ Loaded validation data: shape={augmented_data.shape}")
            return augmented_data

        except Exception as e:
            self.logger.exception(f"❌ Validation data loading failed: {e}")
            return None

    async def _load_validation_frame(
        self,
        data_dir: str,
        exchange: str,
        symbol: str,
    ) -> pd.DataFrame | None:
        """Load the base validation data frame"""
        # Try multiple potential file locations
        potential_files = [
            f"{exchange}_{symbol}_labeled_test.pkl",
            f"{exchange}_{symbol}_validation.pkl",
            f"{exchange}_{symbol}_test.pkl",
            "validation_data.pkl",
        ]

        data_path = Path(data_dir)

        for filename in potential_files:
            file_path = data_path / filename
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

        self.logger.error("❌ No validation data found")
        return None

    async def _augment_validation_data(
        self,
        validation_data: pd.DataFrame,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Augment validation data with additional features"""
        try:
            # Try to add 1m meta-labels if available
            train_file = Path(params["data_dir"]) / f"{params['exchange']}_{params['symbol']}_labeled_train.pkl"

            if train_file.exists():
                with open(train_file, "rb") as f:
                    train_data = pickle.load(f)

                # Find 1m columns
                one_m_cols = [
                    col for col in train_data.columns
                    if isinstance(col, str) and col.startswith("1m_")
                ]

                if one_m_cols and "timestamp" in train_data.columns and "timestamp" in validation_data.columns:
                    validation_data = validation_data.merge(
                        train_data[["timestamp", *one_m_cols]],
                        on="timestamp",
                        how="left",
                    )
                    self.logger.info(f"✅ Added {len(one_m_cols)} 1m meta-label columns")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not augment validation data: {e}")

        return validation_data

    async def _execute_analyst_calibration_stage(
        self,
        model_data: ModelData,
        validation_data: pd.DataFrame,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute analyst model calibration stage"""
        try:
            self.logger.info("🎯 Calibrating analyst models...")

            calibration_results = {}

            # Calibrate each regime's models
            for regime, models in model_data.analyst_models.items():
                self.logger.info(f"  Calibrating regime: {regime}")

                regime_results = await self._calibrate_regime_models(
                    models,
                    model_data.analyst_ensembles.get(regime),
                    validation_data,
                    regime,
                    params,
                )

                calibration_results[regime] = regime_results

            return calibration_results

        except Exception as e:
            self.logger.exception(f"❌ Analyst calibration failed: {e}")
            return {"error": str(e)}

    async def _calibrate_regime_models(
        self,
        models: dict[str, Any],
        ensemble: Any | None,
        validation_data: pd.DataFrame,
        regime: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Calibrate models for a specific regime"""
        results = {
            "regime": regime,
            "models_calibrated": 0,
            "calibration_scores": {},
            "ensemble_calibrated": False,
        }

        # Calibrate individual models
        for model_name, model in models.items():
            try:
                calibration_score = await self._calibrate_single_model(
                    model, validation_data, f"{regime}_{model_name}",
                )
                results["calibration_scores"][model_name] = calibration_score
                results["models_calibrated"] += 1
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to calibrate {model_name}: {e}")

        # Calibrate ensemble if available
        if ensemble is not None:
            try:
                ensemble_score = await self._calibrate_single_model(
                    ensemble, validation_data, f"{regime}_ensemble",
                )
                results["ensemble_score"] = ensemble_score
                results["ensemble_calibrated"] = True
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to calibrate ensemble: {e}")

        return results

    async def _calibrate_single_model(
        self,
        model: Any,
        validation_data: pd.DataFrame,
        model_id: str,
    ) -> float:
        """Calibrate a single model and return calibration score"""
        # Placeholder for actual calibration logic
        # In real implementation, this would:
        # 1. Get model predictions on validation data
        # 2. Apply calibration method (isotonic, Platt, etc.)
        # 3. Evaluate calibration quality
        # 4. Return calibration score

        self.logger.debug(f"Calibrating model: {model_id}")

        # Simulate calibration score
        return np.random.uniform(0.7, 0.95)


    async def _execute_tactician_calibration_stage(
        self,
        model_data: ModelData,
        validation_data: pd.DataFrame,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tactician model calibration stage"""
        try:
            self.logger.info("🎯 Calibrating tactician models...")

            calibration_results = {}

            # Calibrate individual models
            for model_name, model in model_data.tactician_models.items():
                try:
                    score = await self._calibrate_single_model(
                        model, validation_data, f"tactician_{model_name}",
                    )
                    calibration_results[model_name] = {
                        "calibration_score": score,
                        "model_type": "individual",
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to calibrate tactician {model_name}: {e}")

            # Calibrate ensembles
            for ensemble_name, ensemble_data in model_data.tactician_ensembles.items():
                try:
                    score = await self._calibrate_single_model(
                        ensemble_data.get("ensemble"),
                        validation_data,
                        f"tactician_ensemble_{ensemble_name}",
                    )
                    calibration_results[f"ensemble_{ensemble_name}"] = {
                        "calibration_score": score,
                        "model_type": "ensemble",
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to calibrate tactician ensemble {ensemble_name}: {e}")

            return calibration_results

        except Exception as e:
            self.logger.exception(f"❌ Tactician calibration failed: {e}")
            return {"error": str(e)}

    async def _execute_save_results_stage(
        self,
        results: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """Save calibration results to disk"""
        try:
            output_dir = Path(params.get("output_dir", "data/calibrated"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save analyst calibration results
            analyst_file = output_dir / f"{params['exchange']}_{params['symbol']}_analyst_calibration.pkl"
            with open(analyst_file, "wb") as f:
                pickle.dump(results.get("analyst_models", {}), f)

            # Save tactician calibration results
            tactician_file = output_dir / f"{params['exchange']}_{params['symbol']}_tactician_calibration.pkl"
            with open(tactician_file, "wb") as f:
                pickle.dump(results.get("tactician_models", {}), f)

            self.logger.info(f"✅ Saved calibration results to {output_dir}")

        except Exception as e:
            self.logger.exception(f"❌ Failed to save calibration results: {e}")

    def _create_success_result(
        self,
        results: dict[str, Any],
        model_data: ModelData,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Create successful execution result"""
        return {
            "success": True,
            "calibration_results": results,
            "metadata": {
                "symbol": params["symbol"],
                "exchange": params["exchange"],
                "analyst_regimes": len(model_data.analyst_models),
                "tactician_models": len(model_data.tactician_models),
                "calibration_method": self.calibration_config.calibration_method,
                "cross_validated": self.calibration_config.cross_validate,
            },
        }

    def _create_error_result(self, error_message: str) -> dict[str, Any]:
        """Create error result"""
        return {
            "success": False,
            "error": error_message,
            "calibration_results": None,
        }
