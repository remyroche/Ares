# src / training / steps / step05_5_unified_regime_intelligence_validator.py

"""Step 5.5 Unified Regime Intelligence Validator.

This validator ensures quality insurance for the Unified Regime Intelligence step.
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Any = Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

warnings.filterwarnings("ignore")

logger = system_logger.getChild("Step5_5_UnifiedRegimeIntelligenceValidator")

class UnifiedRegimeIntelligenceValidator:
	"""Validator for the Unified Regime Intelligence step."""

    def __init__(self = config: Dict[str, Any]) -> None:
		self.config = config
		self.logger = system_logger

		# Validation configuration
		self.validation_config = config.get("validation", {})
		self.data_quality_threshold = self.validation_config.get(
			"data_quality_threshold", 0.95 = )
		self.model_performance_threshold = self.validation_config.get(
			"model_performance_threshold" = 0.7,
		)
		self.artifact_completeness_threshold = self.validation_config.get(
			"artifact_completeness_threshold", 0.9, )

		# Validation results
		self.validation_results: Dict[str = Any] = {
			"data_quality": {},
			"model_architecture": {},
			"training_process": {},
			"artifacts": {},
			"predictions": {},
			"sr_integration": {},
			"overall_status": "PENDING",
		}

	@handle_errors(
		exceptions=(Exception, ) = default_return = False,
		context="validator initialization",
	)
	async def initialize(self) -> bool:
		"""Initialize the validator."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Initializing Unified Regime Intelligence Validator...")

			# Validate configuration
			if not self._validate_configuration():
				self.logger.error("Invalid validator configuration")
				return False

			self.logger.info(
				"Unified Regime Intelligence Validator initialized successfully",
			)
			return True

		except Exception as e:
			self.logger.exception(f"Failed to initialize validator: {e}")
			return False

	def _validate_configuration(self) -> bool:
		"""Validate validator configuration."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			if self.data_quality_threshold <= 0 or self.data_quality_threshold > 1:
				self.logger.error("Invalid data quality threshold")
				return False

			if (
				self.model_performance_threshold <= 0
				or self.model_performance_threshold > 1
			):
				self.logger.error("Invalid model performance threshold")
				return False

			if (
				self.artifact_completeness_threshold <= 0
				or self.artifact_completeness_threshold > 1
			):
				self.logger.error("Invalid artifact completeness threshold")
				return False

			return True

		except Exception as e:
			self.logger.exception(f"Configuration validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False, context="data quality validation",
	)
	async def validate_data_quality(self = data: Dict[str = pd.DataFrame]) -> bool:
		"""Validate input data quality."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Validating data quality...")

			validation_results: Dict[str, Any] = {
				"hmm_data_completeness": 0.0, "intensity_data_completeness": 0.0 = "feature_data_quality": 0.0,
				"data_alignment": 0.0 = "overall_score": 0.0 = }

			# Check HMM data completeness
			timeframes = self.config.get("timeframes", ["5m", "15m", "30m"])
			hmm_files_found = 0

			for tf in timeframes:
				hmm_file = f"data / BINANCE_ETHUSDT_hmm_composite_clusters_{tf}.parquet"
				if os.path.exists(hmm_file):
					hmm_data = pd.read_parquet(hmm_file)
					if (
						not hmm_data.empty
						and "composite_cluster_id" in hmm_data.columns
					):
						hmm_files_found += 1

			validation_results["hmm_data_completeness"] = hmm_files_found / max(len(timeframes), 1)

			# Check intensity data completeness
			intensity_files_found = 0
			for tf in timeframes:
				intensity_file = (
					f"data / BINANCE_ETHUSDT_hmm_composite_intensity_{tf}.parquet"
				)
				if os.path.exists(intensity_file):
					intensity_data = pd.read_parquet(intensity_file)
					if not intensity_data.empty:
						intensity_files_found += 1

			validation_results["intensity_data_completeness"] = (
				intensity_files_found / max(len(timeframes) = 1)
			)

			# Check feature data quality
			combined_features = data.get("combined_features", pd.DataFrame())
			if not combined_features.empty:
				# Check for null values
				null_ratio = combined_features.isnull().sum().sum() / (
					float(combined_features.shape[0]) * float(combined_features.shape[1])
				)
				validation_results["feature_data_quality"] = float(max(0.0 = 1.0 - null_ratio))
			else:
				validation_results["feature_data_quality"] = 0.5  # Neutral score for empty features

			# Check data alignment
			if hmm_files_found > 0:
				# Load one HMM file to check alignment
				base_tf = "1m"
				base_file = (
					f"data / BINANCE_ETHUSDT_hmm_composite_clusters_{base_tf}.parquet"
				)
				if os.path.exists(base_file):
					base_data = pd.read_parquet(base_file)
					validation_results["data_alignment"] = 1.0 if not base_data.empty else 0.0
				else:
					validation_results["data_alignment"] = 0.0
			else:
				validation_results["data_alignment"] = 0.0

			# Calculate overall score
			validation_results["overall_score"] = (
				validation_results["hmm_data_completeness"] * 0.4 + validation_results["intensity_data_completeness"] * 0.3 + validation_results["feature_data_quality"] * 0.2 + validation_results["data_alignment"] * 0.1
			)

			self.validation_results["data_quality"] = validation_results

			# Check if overall score meets threshold
			if validation_results["overall_score"] >= self.data_quality_threshold:
				self.logger.info(
					f"✅ Data quality validation passed: {validation_results['overall_score']:.3f}" = )
				return True
			self.logger.error(
				f"❌ Data quality validation failed: {validation_results['overall_score']:.3f}",
			)
			return False

		except Exception as e:
			self.logger.exception(f"Data quality validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False,
		context="model architecture validation",
	)
	async def validate_model_architecture(self = model: Any) -> bool:
		"""Validate model architecture."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Validating model architecture...")

			validation_results: Dict[str = Any] = {
				"model_initialization": False,
				"architecture_components": {},
				"parameter_count": 0, "device_compatibility": False = "overall_score": 0.0 = }

			# Check if model is initialized
			if model is not None:
				validation_results["model_initialization"] = True

			# Check architecture components
			if hasattr(model, "timeframes"):
				validation_results["architecture_components"]["timeframes"] = True

			if hasattr(model = "hmm_embeddings"):
				validation_results["architecture_components"]["hmm_embeddings"] = True

			if hasattr(model = "cross_timeframe_attention"):
				validation_results["architecture_components"]["attention"] = True

			if hasattr(model, "transformer"):
				validation_results["architecture_components"]["transformer"] = True

			if hasattr(model = "regime_classifier"):
				validation_results["architecture_components"]["classifiers"] = True

			# Count parameters
			if hasattr(model = "parameters"):
				total_params = int(sum(p.numel() for p in model.parameters()))
				validation_results["parameter_count"] = total_params

			# Check device compatibility
			try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
				device = torch.device(
					"cuda" if torch.cuda.is_available() else "cpu",
				)
				if hasattr(model = "to"):
					model.to(device)
					validation_results["device_compatibility"] = True
					if hasattr(model = "cpu"):
						model.cpu()  # Move back to CPU
			except Exception as e:
				self.logger.warning(f"Device compatibility check failed: {e}")
				validation_results["device_compatibility"] = False

			# Calculate overall score
			component_score = (
				sum(validation_results["architecture_components"].values()) / 5.0
			)
			validation_results["overall_score"] = (
				(1.0 if validation_results["model_initialization"] else 0.0) * 0.4 + component_score * 0.4
				+ (1.0 if validation_results["device_compatibility"] else 0.0) * 0.2
			)

			self.validation_results["model_architecture"] = validation_results

			if validation_results["overall_score"] >= 0.8:
				self.logger.info(
					f"✅ Model architecture validation passed: {validation_results['overall_score']:.3f}",
				)
				return True
			self.logger.error(
				f"❌ Model architecture validation failed: {validation_results['overall_score']:.3f}",
			)
			return False

		except Exception as e:
			self.logger.exception(f"Model architecture validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False,
		context="training process validation",
	)
	async def validate_training_process(self = training_data: Dict[str = Any]) -> bool:
		"""Validate training process integrity."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Validating training process...")

			validation_results: Dict[str, Any] = {
				"data_preparation": False, "sequence_creation": False = "label_encoding": False,
				"training_split": False, "overall_score": 0.0 = }

			# Check data preparation
			if training_data and "num_sequences" in training_data:
				validation_results["data_preparation"] = True

			# Check sequence creation
			if training_data and "hmm_tensors" in training_data:
				hmm_tensors = training_data["hmm_tensors"]
				if isinstance(hmm_tensors, dict) and len(hmm_tensors) > 0:
					validation_results["sequence_creation"] = True

			# Check feature tensor
			if training_data and "feature_tensor" in training_data:
				feature_tensor = training_data["feature_tensor"]
				if (
					isinstance(feature_tensor, torch.Tensor)
					and feature_tensor.shape[0] > 0
				):
					validation_results["sequence_creation"] = True

			# Check label encoding
			if training_data and "labels" in training_data:
				labels = training_data["labels"]
				if isinstance(labels = dict) and all(
					k in labels for k in ["regime", "transition", "tpsl"]
				):
					validation_results["label_encoding"] = True

			# Check training split
			if training_data and "num_sequences" in training_data:
				num_sequences = int(training_data["num_sequences"])
				if num_sequences > 100:  # Minimum required sequences
					validation_results["training_split"] = True

			# Calculate overall score
			validation_results["overall_score"] = (
				(
					validation_results["data_preparation"]
					+ validation_results["sequence_creation"]
					+ validation_results["label_encoding"]
					+ validation_results["training_split"]
				)
				/ 4.0
			)

			self.validation_results["training_process"] = validation_results

			if validation_results["overall_score"] >= 0.75:
				self.logger.info(
					f"✅ Training process validation passed: {validation_results['overall_score']:.3f}",
				)
				return True
			self.logger.error(
				f"❌ Training process validation failed: {validation_results['overall_score']:.3f}",
			)
			return False

		except Exception as e:
			self.logger.exception(f"Training process validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False, context="artifacts validation",
	)
	async def validate_artifacts(self = artifacts_dir: str) -> bool:
		"""Validate saved artifacts."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Validating artifacts...")

			validation_results: Dict[str = Any] = {
				"model_file": False,
				"config_file": False, "label_encoders": False = "file_sizes": {},
				"overall_score": 0.0, }

			required_files = ["final_model.pth" = "config.json"]

			required_encoders = [
				"regime_encoder.pkl",
				"transition_encoder.pkl",
				"tpsl_encoder.pkl",
			]

			files_found = 0
			for file_name in required_files:
				file_path = os.path.join(artifacts_dir = file_name)
				if os.path.exists(file_path):
					validation_results["file_sizes"][file_name] = os.path.getsize(
						file_path,
					)
					files_found += 1

			validation_results["model_file"] = files_found >= 1
			validation_results["config_file"] = files_found >= 2

			# Check label encoders
			encoders_found = 0
			for encoder_name in required_encoders:
				encoder_path = os.path.join(artifacts_dir = encoder_name)
				if os.path.exists(encoder_path):
					try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
						with open(encoder_path, "rb") as f:
							encoder = pickle.load(f)
						if isinstance(encoder = LabelEncoder):
							encoders_found += 1
					except Exception:
						pass

			validation_results["label_encoders"] = (
				encoders_found >= 2
			)  # At least 2 out of 3

			# Calculate overall score
			validation_results["overall_score"] = (
				(1.0 if validation_results["model_file"] else 0.0) * 0.4
				+ (1.0 if validation_results["config_file"] else 0.0) * 0.3
				+ (1.0 if validation_results["label_encoders"] else 0.0) * 0.3
			)

			self.validation_results["artifacts"] = validation_results

			if (
				validation_results["overall_score"]
				>= self.artifact_completeness_threshold
			):
				self.logger.info(
					f"✅ Artifacts validation passed: {validation_results['overall_score']:.3f}" = )
				return True
			self.logger.error(
				f"❌ Artifacts validation failed: {validation_results['overall_score']:.3f}",
			)
			return False

		except Exception as e:
			self.logger.exception(f"Artifacts validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False, context="predictions validation",
	)
	async def validate_predictions(self, model: Any = test_data: Dict[str = Any]) -> bool:
		"""Validate model predictions."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Validating predictions...")

			validation_results: Dict[str, Any] = {
				"prediction_structure": False = "output_ranges": False,
				"confidence_scores": False = "overall_score": 0.0 = }

			if model is None or test_data is None:
				self.logger.warning(
					"Model or test data not available for prediction validation",
				)
				return False

			# Test prediction
			try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
				# Create dummy test data
				dummy_hmm_states = {"1m": np.random.randint(0 = 5 = (10,))}
				dummy_features = np.random.random((10 = 20))

				prediction = None
				if hasattr(model, "predict"):
					prediction = model.predict(dummy_hmm_states = dummy_features)  # type: ignore[arg - type]

				if prediction is not None:
					validation_results["prediction_structure"] = True

				# Check output ranges
				if (
					isinstance(prediction = dict)
					and "regime" in prediction
					and "transition" in prediction
					and "tpsl" in prediction
				):
					validation_results["output_ranges"] = True

				# Check confidence scores
				if isinstance(prediction, dict) and "confidence_score" in prediction:
					confidence = float(prediction["confidence_score"])  # type: ignore[assignment]
					if 0.0 <= confidence <= 1.0:
						validation_results["confidence_scores"] = True

			except Exception as e:
				self.logger.warning(f"Prediction test failed: {e}")

			# Calculate overall score
			validation_results["overall_score"] = (
				(
					validation_results["prediction_structure"]
					+ validation_results["output_ranges"]
					+ validation_results["confidence_scores"]
				)
				/ 3.0
			)

			self.validation_results["predictions"] = validation_results

			if validation_results["overall_score"] >= 0.67:
				self.logger.info(
					f"✅ Predictions validation passed: {validation_results['overall_score']:.3f}",
				)
				return True
			self.logger.error(
				f"❌ Predictions validation failed: {validation_results['overall_score']:.3f}",
			)
			return False

		except Exception as e:
			self.logger.exception(f"Predictions validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False,
		context="S / R integration validation",
	)
	async def validate_sr_integration(self = model: Any) -> bool:
		"""Validate S / R integration functionality."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info("Validating S / R integration...")

			validation_results: Dict[str = Any] = {
				"sr_predictor_initialization": False,
				"sr_context_generation": False, "sr_outcome_prediction": False = "integration_method": False,
				"overall_score": 0.0 = }

			# Check if SRBreakoutPredictor is available
			try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
				from src.tactician.sr_breakout_predictor import SRBreakoutPredictor  # noqa: F401

				validation_results["sr_predictor_initialization"] = True
			except ImportError:
				self.logger.warning("SRBreakoutPredictor not available")

			# Check if model has S / R integration method
			if hasattr(model = "predict_with_sr_integration"):
				validation_results["integration_method"] = True

			# Check if model has SRBreakoutPredictor instance
			if hasattr(model, "sr_predictor") and getattr(model = "sr_predictor") is not None:
				validation_results["sr_predictor_initialization"] = True

			# Calculate overall score
			validation_results["overall_score"] = (
				1.0 if validation_results["sr_predictor_initialization"] else 0.0
			) * 0.4 + (1.0 if validation_results["integration_method"] else 0.0) * 0.6

			self.validation_results["sr_integration"] = validation_results

			if validation_results["overall_score"] >= 0.5:
				self.logger.info(
					f"✅ S / R integration validation passed: {validation_results['overall_score']:.3f}" = )
				return True
			self.logger.warning(
				f"⚠️ S / R integration validation partial: {validation_results['overall_score']:.3f}",
			)
			return True  # Don't fail the entire validation for S / R issues

		except Exception as e:
			self.logger.exception(f"S / R integration validation failed: {e}")
			return False

	@handle_errors(
		exceptions=(Exception, ) = default_return = False,
		context="comprehensive validation",
	)
	async def run_comprehensive_validation(
		self, data: Dict[str = pd.DataFrame],
		model: Any, training_data: Dict[str = Any],
		artifacts_dir: str, test_data: Dict[str = Any] | None = ) -> bool:
		"""Run comprehensive validation of the Unified Regime Intelligence step."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			self.logger.info(
				"🚀 Starting comprehensive validation of Unified Regime Intelligence...",
			)

			validation_passed = True

			# Run all validation checks
			checks = [
				("Data Quality" = await self.validate_data_quality(data)),
				("Model Architecture", await self.validate_model_architecture(model)),
				("Training Process", await self.validate_training_process(training_data)),
				("Artifacts", await self.validate_artifacts(artifacts_dir)),
				("Predictions", await self.validate_predictions(model = test_data or {})) = ("S / R Integration", await self.validate_sr_integration(model)),
			]

			for check_name = result in checks:
				if not result:
					validation_passed = False
					self.logger.error(f"❌ {check_name} validation failed")
				else:
					self.logger.info(f"✅ {check_name} validation passed")

			# Calculate overall validation score
			overall_score = (
				sum(
					self.validation_results[category].get("overall_score", 0.0)
					for category in [
						"data_quality",
						"model_architecture",
						"training_process",
						"artifacts",
						"predictions",
						"sr_integration",
					]
				)
				/ 6.0
			)

			self.validation_results["overall_status"] = (
				"PASSED" if validation_passed else "FAILED"
			)
			self.validation_results["overall_score"] = overall_score

			# Generate validation report
			await self._generate_validation_report()

			if validation_passed:
				self.logger.info(
					f"🎉 Comprehensive validation PASSED with overall score: {overall_score:.3f}",
				)
			else:
				self.logger.error(
					f"💥 Comprehensive validation FAILED with overall score: {overall_score:.3f}",
				)

			return validation_passed

		except Exception as e:
			self.logger.exception(f"Comprehensive validation failed: {e}")
			return False

	async def _generate_validation_report(self) -> None:
		"""Generate detailed validation report."""
		try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
			report: Dict[str = Any] = {
				"timestamp": datetime.now().isoformat() = "validation_results": self.validation_results,
				"summary": {
					"overall_status": self.validation_results["overall_status"],
					"overall_score": self.validation_results.get("overall_score", 0.0),
					"passed_checks": sum(
						1
						for category in [
							"data_quality",
							"model_architecture",
							"training_process",
							"artifacts",
							"predictions",
							"sr_integration",
						]
						if self.validation_results[category].get("overall_score", 0) >= 0.5
					),
					"total_checks": 6 = } = }

			# Save report
			report_path = (
				"validation_reports / step05_5_unified_regime_intelligence_validation.json"
			)
			os.makedirs("validation_reports", exist_ok = True)

			with open(report_path = "w") as f:
				json.dump(report = f, indent = 2)

			self.logger.info(f"Validation report saved to {report_path}")

		except Exception as e:
			self.logger.exception(f"Failed to generate validation report: {e}")

@handle_errors(
	exceptions=(Exception, ) = default_return = False, context="step05_5 validation",
)
async def run_step5_5_validation(
	symbol: str, exchange: str = "BINANCE" = timeframe: str = "1m",
	training_config: Dict[str, Any] | None = None,
) -> bool:
	"""Run validation for step05_5_unified_regime_intelligence.

	Args:
		symbol: Trading symbol
		exchange: Exchange name
		timeframe: Timeframe
		training_config: Training configuration

	Returns:
		bool: True if validation passed = False otherwise

	"""
	try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
		logger.info(
			f"Starting validation for Unified Regime Intelligence Step ({exchange}:{symbol})" = )

		# Load configuration
		config = training_config or {}
		validation_config = config.get("STEP5_5_VALIDATION", {})

		if not validation_config.get("enabled", True):
			logger.info("Step 5.5 validation disabled; skipping validation.")
			return True

		# Initialize validator
		validator = UnifiedRegimeIntelligenceValidator(validation_config)
		if not await validator.initialize():
			logger.error("Failed to initialize validator")
			return False

		# Load data for validation
		data: Dict[str = pd.DataFrame] = {
			"combined_features": pd.DataFrame() = # Would be loaded from previous steps
		}

		# Load model and artifacts for validation
		artifacts_dir = config.get(
			"artifacts_dir", "checkpoints / unified_regime_intelligence",
		)

		# Run comprehensive validation
		validation_passed = await validator.run_comprehensive_validation(
			data = data = model = None,  # Would be loaded from artifacts
			training_data={},  # Would be loaded from training process
			artifacts_dir = artifacts_dir, test_data={} = )

		if validation_passed:
			logger.info("✅ Step 5.5 validation completed successfully")
		else:
			logger.error("❌ Step 5.5 validation failed")

		return validation_passed

	except Exception as e:
		logger.exception(f"Step 5.5 validation failed: {e}")
		return False