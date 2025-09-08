
import os
import json
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common_operations import safe_json_load
from src.utils.logger import system_logger
from src.utils.warning_symbols import failed, missing, success, warning, error

from ...core.decorators import handles_errors
from ..standardized_parquet_handler import standardized_parquet_handler

# src/training/steps/step13_*.py

logger = system_logger

class Step7AnalystEnsembleCreationValidator:
    """Validator for Step 7: Analyst Ensemble Creation with enhanced validation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger
        self.validation_results = {}
        self.validation_metrics = {
            'start_time': None,
            'validation_duration': 0.0,
            'files_checked': 0,
            'models_validated': 0,
            'errors_found': 0,
            'warnings_found': 0
        }

    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst ensemble creation validation",
    )
    def validate(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        training_input: dict[str, Any],
    ) -> bool:
        """Validate Step 7: Analyst Ensemble Creation with comprehensive checks.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes

        """
        self.validation_metrics['start_time'] = time.time()
        self.logger.info("🔍 Starting Step 7: Analyst Ensemble Creation validation")

        try:
            # Fast fail validation
            if not self._fast_fail_validation(symbol, exchange, data_dir, training_input):
                self.logger.error("❌ Fast fail validation failed")
                return False

            # Validate ensemble files exist
            ensemble_files_passed = self._validate_ensemble_files(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
            )
            self.validation_results["ensemble_files"] = ensemble_files_passed

            # Validate ensemble structure
            ensemble_structure_passed = self._validate_ensemble_structure(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
            )
            self.validation_results["ensemble_structure"] = ensemble_structure_passed

            # Validate ensemble weights
            ensemble_weights_passed = self._validate_ensemble_weights(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
            )
            self.validation_results["ensemble_weights"] = ensemble_weights_passed

            # Validate model quality
            model_quality_passed = self._validate_model_quality(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
            )
            self.validation_results["model_quality"] = model_quality_passed

            # Validate data integrity
            data_integrity_passed = self._validate_data_integrity(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
            )
            self.validation_results["data_integrity"] = data_integrity_passed

            # Overall validation result
            overall_passed = all([
                ensemble_files_passed,
                ensemble_structure_passed,
                ensemble_weights_passed,
                model_quality_passed,
                data_integrity_passed
            ])

            # Log validation metrics
            self._log_validation_metrics()

            if overall_passed:
                self.logger.info("✅ Step 7: Analyst Ensemble Creation validation passed")
                self.print_message(
                    success("✅ Step 7: Analyst Ensemble Creation validation passed"),
                )
            else:
                self.logger.warning("⚠️ Step 7: Analyst Ensemble Creation validation failed")
                self.print_message(
                    failed("⚠️ Step 7: Analyst Ensemble Creation validation failed"),
                )

            return overall_passed

        except Exception as e:
            self.logger.exception(f"❌ Error in Step 7 validation: {e}")
            self.print_message(error(f"❌ Error in Step 7 validation: {e}"))
            return False

    def _fast_fail_validation(self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]) -> bool:
        """Fast fail validation for early error detection."""
        try:
            # Check basic parameters
            if not symbol or not exchange or not data_dir:
                self.logger.error("❌ Missing required parameters: symbol, exchange, or data_dir")
                return False
            
            # Check data directory accessibility
            if not os.path.exists(data_dir):
                self.logger.error(f"❌ Data directory does not exist: {data_dir}")
                return False
            
            if not os.access(data_dir, os.R_OK):
                self.logger.error(f"❌ Data directory is not readable: {data_dir}")
                return False
            
            # Check training input structure
            if not isinstance(training_input, dict):
                self.logger.error("❌ Training input must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fast fail validation error: {e}")
            return False
    
    def _validate_ensemble_files(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate that ensemble files exist with enhanced checks."""
        try:
            # Expected ensemble files
            ensemble_dir = os.path.join(data_dir, "analyst_ensemble")
            summary_file = os.path.join(
                ensemble_dir,
                f"{exchange}_{symbol}_analyst_ensemble_summary.json",
            )

            missing_files = []
            file_issues = []

            # Check if ensemble directory exists
            if not os.path.isdir(ensemble_dir):
                missing_files.append(ensemble_dir)
            else:
                self.validation_metrics['files_checked'] += 1

            # Check if summary file exists
            if not os.path.isfile(summary_file):
                missing_files.append(summary_file)
            else:
                self.validation_metrics['files_checked'] += 1
                
                # Check file size and permissions
                try:
                    file_size = os.path.getsize(summary_file)
                    if file_size < 100:  # Less than 100 bytes is suspicious
                        file_issues.append(f"Summary file too small: {file_size} bytes")
                    
                    if not os.access(summary_file, os.R_OK):
                        file_issues.append("Summary file not readable")
                        
                except Exception as e:
                    file_issues.append(f"Error checking summary file: {e}")

            if missing_files:
                self.logger.error(
                    missing(f"❌ Missing ensemble files: {missing_files}"),
                )
                self.print_message(missing(f"❌ Missing ensemble files: {missing_files}"))
                self.validation_metrics['errors_found'] += len(missing_files)
                return False
            
            if file_issues:
                for issue in file_issues:
                    self.logger.warning(f"⚠️ {issue}")
                    self.validation_metrics['warnings_found'] += 1

            self.logger.info("✅ All ensemble files exist and are accessible")
            return True

        except Exception as e:
            self.logger.exception(error(f"❌ Error validating ensemble files: {e}"))
            self.validation_metrics['errors_found'] += 1
            return False

    def _validate_ensemble_structure(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate ensemble structure and metadata with enhanced checks."""
        try:
            summary_file = os.path.join(
                data_dir,
                "analyst_ensemble",
                f"{exchange}_{symbol}_analyst_ensemble_summary.json",
            )

            if not os.path.exists(summary_file):
                self.logger.error(
                    missing(f"❌ Ensemble summary file not found: {summary_file}"),
                )
                self.validation_metrics['errors_found'] += 1
                return False

            # Load and validate summary
            summary = safe_json_load(summary_file)
            if summary is None:
                self.logger.error("❌ Failed to load ensemble summary file")
                self.validation_metrics['errors_found'] += 1
                return False

            # Check required fields
            required_fields = [
                "ensemble_models",
                "ensemble_weights",
                "ensemble_metadata",
            ]
            missing_fields = [
                field for field in required_fields if field not in summary
            ]

            if missing_fields:
                self.logger.error(
                    failed(
                        f"❌ Missing required fields in ensemble summary: {missing_fields}",
                    ),
                )
                self.print_message(
                    failed(
                        f"❌ Missing required fields in ensemble summary: {missing_fields}",
                    ),
                )
                self.validation_metrics['errors_found'] += len(missing_fields)
                return False

            # Validate metadata
            metadata = summary["ensemble_metadata"]
            if not isinstance(metadata, dict):
                self.logger.error("❌ Ensemble metadata must be a dictionary")
                self.validation_metrics['errors_found'] += 1
                return False
            
            if metadata.get("symbol") != symbol or metadata.get("exchange") != exchange:
                self.logger.error(
                    failed(
                        f"❌ Metadata mismatch: expected {exchange}_{symbol}, got {metadata.get('exchange')}_{metadata.get('symbol')}",
                    ),
                )
                self.print_message(
                    failed(
                        f"❌ Metadata mismatch: expected {exchange}_{symbol}, got {metadata.get('exchange')}_{metadata.get('symbol')}",
                    ),
                )
                self.validation_metrics['errors_found'] += 1
                return False

            # Validate model count
            model_count = metadata.get("model_count", 0)
            if not isinstance(model_count, int) or model_count < 0:
                self.logger.error(f"❌ Invalid model count: {model_count}")
                self.validation_metrics['errors_found'] += 1
                return False
            
            if model_count == 0:
                self.logger.warning("⚠️ No models in ensemble")
                self.validation_metrics['warnings_found'] += 1

            # Check if it's a placeholder ensemble
            if metadata.get("is_placeholder", False):
                self.logger.warning(
                    warning(
                        "⚠️ Ensemble is a placeholder (no enhanced models from Step 6)",
                    ),
                )
                self.print_message(
                    warning(
                        "⚠️ Ensemble is a placeholder (no enhanced models from Step 6)",
                    ),
                )
                self.validation_metrics['warnings_found'] += 1

            self.logger.info("✅ Ensemble structure validation passed")
            return True

        except Exception as e:
            self.logger.exception(error(f"❌ Error validating ensemble structure: {e}"))
            self.validation_metrics['errors_found'] += 1
            return False

    def _validate_ensemble_weights(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate ensemble weights."""
        try:
            summary_file = os.path.join(
                data_dir,
                "analyst_ensemble",
                f"{exchange}_{symbol}_analyst_ensemble_summary.json",
            )

            if not os.path.exists(summary_file):
                return False

            summary = safe_json_load(summary_file)
            if summary is None:
                return False

            ensemble_weights = summary.get("ensemble_weights", {})
            if not isinstance(ensemble_weights, dict):
                self.logger.error("❌ Ensemble weights must be a dictionary")
                self.validation_metrics['errors_found'] += 1
                return False

            # Validate weights for each regime
            for regime, weights in ensemble_weights.items():
                if not isinstance(weights, dict):
                    self.logger.error(f"❌ Weights for regime {regime} must be a dictionary")
                    self.validation_metrics['errors_found'] += 1
                    return False
                
                # Check weight sum is approximately 1.0
                weight_sum = sum(weights.values())
                if not np.isclose(weight_sum, 1.0, atol=1e-6):
                    self.logger.error(f"❌ Weights for regime {regime} sum to {weight_sum}, not 1.0")
                    self.validation_metrics['errors_found'] += 1
                    return False
                
                # Check all weights are non-negative
                if any(w < 0 for w in weights.values()):
                    self.logger.error(f"❌ Negative weights found in regime {regime}")
                    self.validation_metrics['errors_found'] += 1
                    return False
                
                # Check for NaN or infinite values
                if any(not np.isfinite(w) for w in weights.values()):
                    self.logger.error(f"❌ Non-finite weights found in regime {regime}")
                    self.validation_metrics['errors_found'] += 1
                    return False
            
            self.logger.info("✅ Ensemble weights validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(error(f"❌ Error validating ensemble weights: {e}"))
            self.validation_metrics['errors_found'] += 1
            return False
    
    def _validate_model_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate model quality and diversity."""
        try:
            summary_file = os.path.join(
                data_dir,
                "analyst_ensemble",
                f"{exchange}_{symbol}_analyst_ensemble_summary.json",
            )

            if not os.path.exists(summary_file):
                return False

            summary = safe_json_load(summary_file)
            if summary is None:
                return False

            ensemble_models = summary.get("ensemble_models", {})
            if not isinstance(ensemble_models, dict):
                self.logger.error("❌ Ensemble models must be a dictionary")
                self.validation_metrics['errors_found'] += 1
                return False

            # Check model diversity
            all_model_names = []
            for regime, models in ensemble_models.items():
                if isinstance(models, dict):
                    all_model_names.extend(models.keys())
            
            if len(set(all_model_names)) < len(all_model_names):
                self.logger.warning("⚠️ Duplicate model names found across regimes")
                self.validation_metrics['warnings_found'] += 1
            
            # Check for minimum model count
            total_models = len(all_model_names)
            if total_models < 2:
                self.logger.warning(f"⚠️ Only {total_models} models in ensemble (minimum 2 recommended)")
                self.validation_metrics['warnings_found'] += 1
            
            self.validation_metrics['models_validated'] = total_models
            self.logger.info(f"✅ Model quality validation passed ({total_models} models)")
            return True
            
        except Exception as e:
            self.logger.exception(error(f"❌ Error validating model quality: {e}"))
            self.validation_metrics['errors_found'] += 1
            return False
    
    def _validate_data_integrity(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate data integrity and consistency."""
        try:
            summary_file = os.path.join(
                data_dir,
                "analyst_ensemble",
                f"{exchange}_{symbol}_analyst_ensemble_summary.json",
            )

            if not os.path.exists(summary_file):
                return False

            summary = safe_json_load(summary_file)
            if summary is None:
                return False

            # Check timestamp format
            metadata = summary.get("ensemble_metadata", {})
            created_at = metadata.get("created_at")
            if created_at:
                try:
                    pd.to_datetime(created_at)
                except Exception:
                    self.logger.warning(f"⚠️ Invalid timestamp format: {created_at}")
                    self.validation_metrics['warnings_found'] += 1
            
            # Check for required metadata fields
            required_metadata_fields = ['symbol', 'exchange', 'created_at', 'model_count']
            missing_metadata = [field for field in required_metadata_fields if field not in metadata]
            
            if missing_metadata:
                self.logger.warning(f"⚠️ Missing metadata fields: {missing_metadata}")
                self.validation_metrics['warnings_found'] += len(missing_metadata)
            
            self.logger.info("✅ Data integrity validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(error(f"❌ Error validating data integrity: {e}"))
            self.validation_metrics['errors_found'] += 1
            return False
    
    def _log_validation_metrics(self):
        """Log validation metrics."""
        try:
            if self.validation_metrics['start_time']:
                self.validation_metrics['validation_duration'] = time.time() - self.validation_metrics['start_time']
            
            self.logger.info("📊 Validation Metrics:")
            self.logger.info(f"   ⏱️ Duration: {self.validation_metrics['validation_duration']:.2f}s")
            self.logger.info(f"   📁 Files checked: {self.validation_metrics['files_checked']}")
            self.logger.info(f"   🤖 Models validated: {self.validation_metrics['models_validated']}")
            self.logger.info(f"   ❌ Errors found: {self.validation_metrics['errors_found']}")
            self.logger.info(f"   ⚠️ Warnings found: {self.validation_metrics['warnings_found']}")
            
        except Exception as e:
            self.logger.debug(f"Error logging validation metrics: {e}")
    
    def print_message(self, message: str) -> None:
        """Print validation message."""
        self.logger.info(message)

def step7_analyst_ensemble_creation_validator(
    symbol: str,
    exchange: str,
    data_dir: str,
    training_input: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Step 7: Analyst Ensemble Creation Validator.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if validation passes

    """
    validator = Step7AnalystEnsembleCreationValidator(config)
    return validator.validate(symbol, exchange, data_dir, training_input)
