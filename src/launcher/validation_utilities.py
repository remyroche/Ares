#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Validation Utilities for Ares Launcher

This module contains validation utilities that extract validation logic
from the main launcher class, improving code organization and reusability.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

from src.utils.common_operations import safe_file_exists, ensure_directory

class BaseValidator:
    """Base class for all validators."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, **kwargs) -> bool:
        """Perform validation and return success status."""
        self.logger.info("🔍 Performing base validation...")
        try:
            # Basic validation - check if logger is working
            if not self.logger:
                self.logger.error("❌ Logger not available")
                return False

            self.logger.info("✅ Base validation completed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Base validation failed: {e}")
            return False

class PrerequisitesValidator(BaseValidator):
    """Validates prerequisites for various operations."""

    def validate(self, operation_type: str, symbol: str, exchange: str, **kwargs) -> bool:
        """Validate prerequisites for the specified operation."""
        validators = {
            "data_collection": self._validate_data_collection_prerequisites,
            "model_training": self._validate_model_training_prerequisites,
            "optimisation": self._validate_optimisation_prerequisites,
            "backtesting": self._validate_backtesting_prerequisites,
        }

        if operation_type not in validators:
            self.logger.error(f"Unknown operation type for validation: {operation_type}")
            return False

        return validators[operation_type](symbol, exchange, **kwargs)

    def _validate_data_collection_prerequisites(self, symbol: str, exchange: str, **kwargs) -> bool:
        """Validate prerequisites for data collection."""
        self.logger.info("🔍 Validating data collection prerequisites...")
        tprint("🔍 Validating data collection prerequisites...")

        try:
            # Check required directories
            required_dirs = ["data_cache", "log"]

            for dir_path in required_dirs:
                if not safe_file_exists(dir_path):
                    self.logger.warning(f"⚠️ Creating missing directory: {dir_path}")
                    ensure_directory(dir_path)
                else:
                    self.logger.info(f"✅ Directory exists: {dir_path}")

            # Check for standalone data collection script
            script_path = "standalone_data_collection.py"
            if not safe_file_exists(script_path):
                self.logger.error(f"❌ Data collection script not found: {script_path}")
                tprint(f"❌ Data collection script not found: {script_path}")
                return False
            else:
                self.logger.info(f"✅ Data collection script found: {script_path}")

            self.logger.info("✅ Data collection prerequisites validation completed")
            tprint("✅ Data collection prerequisites validation completed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            tprint(f"❌ Prerequisites validation failed: {e}")
            return False

    def _validate_model_training_prerequisites(self, symbol: str, exchange: str, **kwargs) -> bool:
        """Validate prerequisites for model training."""
        self.logger.info("🔍 Validating model training prerequisites...")
        tprint("🔍 Validating model training prerequisites...")

        try:
            # Check required directories
            required_dirs = ["data_cache", "models", "checkpoints", "log"]

            for dir_path in required_dirs:
                if not safe_file_exists(dir_path):
                    self.logger.warning(f"⚠️ Creating missing directory: {dir_path}")
                    ensure_directory(dir_path)
                else:
                    self.logger.info(f"✅ Directory exists: {dir_path}")

            # Check for required data files
            required_data_files = [
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"data_cache/volume_{exchange}_{symbol}_consolidated.parquet"
            ]

            missing_files = []
            for file_path in required_data_files:
                if not safe_file_exists(file_path):
                    missing_files.append(file_path)
                else:
                    self.logger.info(f"✅ Data file exists: {file_path}")

            if missing_files:
                self.logger.error(f"❌ Missing required data files: {missing_files}")
                tprint(f"❌ Missing required data files:")
                for file_path in missing_files:
                    tprint(f"   • {file_path}")
                tprint("💡 Please run data collection first:")
                tprint(f"   python ares_launcher.py load --symbol {symbol} --exchange {exchange}")
                return False

            # Check for previous step outputs
            previous_step_files = [
                f"data_cache/features_{exchange}_{symbol}_consolidated.parquet",
                f"data_cache/labels_{exchange}_{symbol}_consolidated.parquet"
            ]

            missing_previous = []
            for file_path in previous_step_files:
                if not safe_file_exists(file_path):
                    missing_previous.append(file_path)

            if missing_previous:
                self.logger.warning(f"⚠️ Some previous step outputs missing: {missing_previous}")
                tprint("⚠️ Some previous step outputs are missing - model training will use defaults")

            self.logger.info("✅ Model training prerequisites validation completed")
            tprint("✅ Model training prerequisites validation completed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            tprint(f"❌ Prerequisites validation failed: {e}")
            return False

    def _validate_optimisation_prerequisites(self, symbol: str, exchange: str, **kwargs) -> bool:
        """Validate prerequisites for optimisation."""
        self.logger.info("🔍 Validating optimisation prerequisites...")
        tprint("🔍 Validating optimisation prerequisites...")

        try:
            # Check required directories
            required_dirs = ["data_cache", "models", "checkpoints", "log"]

            for dir_path in required_dirs:
                if not safe_file_exists(dir_path):
                    self.logger.warning(f"⚠️ Creating missing directory: {dir_path}")
                    ensure_directory(dir_path)
                else:
                    self.logger.info(f"✅ Directory exists: {dir_path}")

            # Check for required data files
            required_data_files = [
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"data_cache/volume_{exchange}_{symbol}_consolidated.parquet"
            ]

            missing_files = []
            for file_path in required_data_files:
                if not safe_file_exists(file_path):
                    missing_files.append(file_path)
                else:
                    self.logger.info(f"✅ Data file exists: {file_path}")

            if missing_files:
                self.logger.error(f"❌ Missing required data files: {missing_files}")
                tprint(f"❌ Missing required data files:")
                for file_path in missing_files:
                    tprint(f"   • {file_path}")
                tprint("💡 Please run data collection first:")
                tprint(f"   python ares_launcher.py load --symbol {symbol} --exchange {exchange}")
                return False

            # Check for previous step outputs
            previous_step_files = [
                f"models/{symbol}_{exchange}_tactician_specialist.pkl",
                f"models/{symbol}_{exchange}_confidence_calibration.json"
            ]

            missing_previous = []
            for file_path in previous_step_files:
                if not safe_file_exists(file_path):
                    missing_previous.append(file_path)

            if missing_previous:
                self.logger.warning(f"⚠️ Some previous step outputs missing: {missing_previous}")
                tprint("⚠️ Some previous step outputs are missing - optimisation will use defaults")

            self.logger.info("✅ Optimisation prerequisites validation completed")
            tprint("✅ Optimisation prerequisites validation completed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            tprint(f"❌ Prerequisites validation failed: {e}")
            return False

    def _validate_backtesting_prerequisites(self, symbol: str, exchange: str, **kwargs) -> bool:
        """Validate prerequisites for backtesting."""
        self.logger.info("🔍 Validating backtesting prerequisites...")
        tprint("🔍 Validating backtesting prerequisites...")

        try:
            # Check required directories
            required_dirs = ["data_cache", "models", "log"]

            for dir_path in required_dirs:
                if not safe_file_exists(dir_path):
                    self.logger.warning(f"⚠️ Creating missing directory: {dir_path}")
                    ensure_directory(dir_path)
                else:
                    self.logger.info(f"✅ Directory exists: {dir_path}")

            # Check for required data files
            required_data_files = [
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"data_cache/volume_{exchange}_{symbol}_consolidated.parquet"
            ]

            missing_files = []
            for file_path in required_data_files:
                if not safe_file_exists(file_path):
                    missing_files.append(file_path)
                else:
                    self.logger.info(f"✅ Data file exists: {file_path}")

            if missing_files:
                self.logger.error(f"❌ Missing required data files: {missing_files}")
                tprint(f"❌ Missing required data files:")
                for file_path in missing_files:
                    tprint(f"   • {file_path}")
                tprint("💡 Please run data collection first:")
                tprint(f"   python ares_launcher.py load --symbol {symbol} --exchange {exchange}")
                return False

            self.logger.info("✅ Backtesting prerequisites validation completed")
            tprint("✅ Backtesting prerequisites validation completed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            tprint(f"❌ Prerequisites validation failed: {e}")
            return False

class StepValidationValidator(BaseValidator):
    """Validates step dependencies and prerequisites."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.step_dependencies = self._build_step_dependencies()

    def validate(self, start_step: str, symbol: str, exchange: str, **kwargs) -> bool:
        """Validate all previous steps before starting from a specific step."""
        self.logger.info(f"🔍 Validating previous steps before starting from {start_step}")

        try:
            from src.utils.validator_orchestrator import ValidatorOrchestrator

            # Create validator orchestrator
            validator_orchestrator = ValidatorOrchestrator()

            # Prepare training input for validation
            timeframe = kwargs.get('timeframe', '15m')
            if timeframe not in {'15m', '5m'}:
                self.logger.info(
                    f"ℹ️ Using non-standard timeframe '{timeframe}' for validation."
                )
                tprint(
                    f"ℹ️ [STEP_VALIDATION] Using caller-specified timeframe: {timeframe}"
                )
            elif timeframe == '15m':
                tprint(
                    "ℹ️ [STEP_VALIDATION] Defaulting to 15m Analyst timeframe; override to 5m for Tactician steps."
                )
            else:  # timeframe == '5m'
                tprint(
                    "ℹ️ [STEP_VALIDATION] Validating with 5m Tactician timeframe (requires Analyst green-signal filtered data)."
                )

            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "data_dir": "data_cache",
            }

            # Get step dependencies
            steps_to_validate = self._get_required_steps(start_step)

            if not steps_to_validate:
                self.logger.info(f"✅ No previous steps to validate for {start_step}")
                return True

            self.logger.info(f"🔍 Validating {len(steps_to_validate)} previous steps: {steps_to_validate}")

            # Validate each required step
            validation_results = {}
            all_passed = True

            for step in steps_to_validate:
                self.logger.info(f"🔍 Validating {step}...")
                try:
                    result = validator_orchestrator.run_step_validator(
                        step, training_input, {}, kwargs.get('config', {})
                    )
                    validation_results[step] = result

                    if result.get("validation_passed", False):
                        self.logger.info(f"✅ {step} validation passed")
                    else:
                        self.logger.error(f"❌ {step} validation failed: {result.get('error', 'Unknown error')}")
                        all_passed = False

                except Exception as e:
                    self.logger.exception(f"❌ Error validating {step}: {e}")
                    validation_results[step] = {"validation_passed": False, "error": str(e)}
                    all_passed = False

            # Print validation report
            self._print_validation_report(validation_results, symbol, exchange, start_step)

            return all_passed

        except Exception as e:
            self.logger.exception(f"❌ Error in step validation: {e}")
            return False

    def _build_step_dependencies(self) -> Dict[str, List[str]]:
        """Build the step dependency graph."""
        return {
            "step1_data_collection": [],
            "step1_5_data_converter": ["step1_data_collection"],
            "step2_data_reading": ["step1_data_collection", "step1_5_data_converter"],
            "step3_5_final_regime_clustering": [],
            "step4_triple_barrier_method": ["step3_5_final_regime_clustering"],
            "step4_regime_data_splitting": ["step4_triple_barrier_method"],
            "step5_labeling": ["step4_regime_data_splitting"],
            "step6_feature_engineering": ["step5_labeling"],
            "step7_enhanced_matrix_operations": ["step6_feature_engineering"],
            "step8_regime_data_splitting": ["step7_enhanced_matrix_operations"],
            "step9_hmm_based_training": ["step8_regime_data_splitting"],
            "step9_5_hmm_lm_generalist_training": ["step9_hmm_based_training"],
            "step10_unified_regime_intelligence": ["step9_5_hmm_lm_generalist_training"],
            "step11_analyst_creation": ["step10_unified_regime_intelligence"],
            "step12_analyst_enhancement": ["step11_analyst_creation"],
            "step13_analyst_ensemble_creation": ["step12_analyst_enhancement"],
            "step14_tactician_labeling": ["step13_analyst_ensemble_creation"],
            "step15_tactician_specialist_training": ["step14_tactician_labeling"],
            "step16_confidence_calibration": ["step15_tactician_specialist_training"],
            "step17_final_parameters_optimization": ["step16_confidence_calibration"],
            "step18_walk_forward_validation": ["step17_final_parameters_optimization"],
            "step19_monte_carlo_validation": ["step18_walk_forward_validation"],
            "step20_ab_testing": ["step19_monte_carlo_validation"],
            "step21_saving": ["step20_ab_testing"],
        }

    def _get_required_steps(self, start_step: str) -> List[str]:
        """Get all steps that need to be validated before starting from a specific step."""
        required_steps = []

        # Use a simple approach: validate all steps that come before the start step
        step_order = [
            "step1_data_collection",
            "step1_5_data_converter",
            "step2_data_reading",
            "step3_5_final_regime_clustering",
            "step4_triple_barrier_method",
            "step4_regime_data_splitting",
            "step5_labeling",
            "step6_feature_engineering",
            "step7_enhanced_matrix_operations",
            "step8_regime_data_splitting",
            "step9_hmm_based_training",
            "step9_5_hmm_lm_generalist_training",
            "step10_unified_regime_intelligence",
            "step11_analyst_creation",
            "step12_analyst_enhancement",
            "step13_analyst_ensemble_creation",
            "step14_tactician_labeling",
            "step15_tactician_specialist_training",
            "step16_confidence_calibration",
            "step17_final_parameters_optimization",
            "step18_walk_forward_validation",
            "step19_monte_carlo_validation",
            "step20_ab_testing",
            "step21_saving",
        ]

        try:
            start_index = step_order.index(start_step)
            required_steps = step_order[:start_index]
        except ValueError:
            self.logger.warning(f"⚠️ Unknown step {start_step}, skipping validation")
            return []

        return required_steps

    def _print_validation_report(self, validation_results: Dict, symbol: str, exchange: str, start_step: str):
        """Print a formatted validation report."""
        tprint("\n" + "="*80)
        tprint("📊 STEP VALIDATION REPORT")
        tprint(f"🎯 Symbol: {symbol}")
        tprint(f"🏢 Exchange: {exchange}")
        tprint(f"🚀 Starting from: {start_step}")
        tprint("="*80)

        all_passed = True
        for step, result in validation_results.items():
            passed = result.get("validation_passed", False)
            status = "✅ PASSED" if passed else "❌ FAILED"
            tprint(f"{step:<35} {status}")

            if not passed:
                all_passed = False
                error = result.get("error", "Unknown error")
                tprint(f"   Error: {error}")

        tprint("="*80)
        if all_passed:
            tprint("🎉 All previous steps validated successfully!")
        else:
            tprint("❌ Some previous steps failed validation")
        tprint("="*80)

class DataValidationValidator(BaseValidator):
    """Validates data for step02 readiness."""

    def validate(self, symbol: str, exchange: str, **kwargs) -> Tuple[bool, Dict]:
        """Validate data for step02 readiness."""
        self.logger.info(f"🔍 Validating data for step02 readiness for {symbol} on {exchange}")

        try:

            # Create validator orchestrator
            validator_orchestrator = ValidatorOrchestrator()

            # Prepare training input for validation
            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "1m",
                "data_dir": "data_cache",
            }

            # Empty pipeline state since we're checking existing data
            pipeline_state = {}

            # Validate step01 and step1_5 using existing validators
            self.logger.info("🔍 Validating step1_data_collection using existing validator")
            step1_result = validator_orchestrator.run_step_validator(
                "step1_data_collection", training_input, pipeline_state, kwargs.get('config', {})
            )

            self.logger.info("🔍 Validating step1_5_data_converter using existing validator")
            step1_5_result = validator_orchestrator.run_step_validator(
                "step1_5_data_converter", training_input, pipeline_state, kwargs.get('config', {})
            )

            # Print validation report
            self._print_step2_validation_report(step1_result, step1_5_result, symbol, exchange)

            # Check if we can proceed
            step1_passed = step1_result.get("validation_passed", False)
            step1_5_passed = step1_5_result.get("validation_passed", False)
            can_start = step1_passed and step1_5_passed

            if not can_start:
                self.logger.error("❌ Cannot start from step02 - data validation failed")
                self.logger.error("Please run step01 and step1_5 first to collect and process data")
                return False, {}

            # Log warnings if any issues found
            step1_warnings = step1_result.get("warnings", [])
            step1_5_warnings = step1_5_result.get("warnings", [])
            total_warnings = len(step1_warnings) + len(step1_5_warnings)

            if total_warnings > 0:
                self.logger.warning(f"⚠️ Data validation found {total_warnings} warnings - proceeding with existing data")
                for warning in step1_warnings:
                    self.logger.warning(f"   • Step1: {warning}")
                for warning in step1_5_warnings:
                    self.logger.warning(f"   • Step1_5: {warning}")

            self.logger.info("✅ Data validation passed - proceeding with existing data")

            return True, {
                "step1_result": step1_result,
                "step1_5_result": step1_5_result,
                "warnings": step1_warnings + step1_5_warnings
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Could not run existing validators: {e}")
            self.logger.warning("Proceeding with basic file existence check")

            # Fallback to basic check - look for unified data
            unified_data_dir = f"data/training/unified/{exchange.lower()}/{symbol}/1m/exchange={exchange.upper()}"
            if not os.path.exists(unified_data_dir):
                self.logger.error(f"❌ Unified data directory not found: {unified_data_dir}")
                self.logger.error("Please run data loading first or ensure unified data exists")
                return False, {}

            # Check if there are any parquet files in the unified data directory
            parquet_files = []
            for root, dirs, files in os.walk(unified_data_dir):
                parquet_files.extend([f for f in files if f.endswith('.parquet')])

            if not parquet_files:
                self.logger.error(f"❌ No parquet files found in unified data directory: {unified_data_dir}")
                self.logger.error("Please run data loading first or ensure unified data exists")
                return False, {}

            self.logger.info(f"✅ Found unified data: {unified_data_dir} ({len(parquet_files)} parquet files)")

            return True, {"unified_data_dir": unified_data_dir, "parquet_files": parquet_files}

    def _print_step2_validation_report(self, step1_result: Dict, step1_5_result: Dict, symbol: str, exchange: str):
        """Print a formatted validation report for step02 readiness."""
        tprint("\n" + "="*80)
        tprint("📊 DATA VALIDATION REPORT FOR STEP2")
        tprint(f"🎯 Symbol: {symbol}")
        tprint(f"🏢 Exchange: {exchange}")
        tprint("="*80)

        # Step1 status
        step1_passed = step1_result.get("validation_passed", False)
        step1_status = "✅ PASSED" if step1_passed else "❌ FAILED"
        step1_warnings = step1_result.get("warnings", [])
        tprint(f"📁 Step1 Data Collection: {step1_status}")
        if step1_warnings:
            tprint(f"   ⚠️  Found {len(step1_warnings)} warnings")
            for warning in step1_warnings:
                tprint(f"     • {warning}")

        # Step1_5 status
        step1_5_passed = step1_5_result.get("validation_passed", False)
        step1_5_status = "✅ PASSED" if step1_5_passed else "❌ FAILED"
        step1_5_warnings = step1_5_result.get("warnings", [])
        tprint(f"🔄 Step1_5 Data Converter: {step1_5_status}")
        if step1_5_warnings:
            tprint(f"   ⚠️  Found {len(step1_5_warnings)} warnings")
            for warning in step1_5_warnings:
                tprint(f"     • {warning}")

        # Show validation details if available
        if step1_result.get("details"):
            tprint(f"   📋 Step1 Details: {step1_result['details']}")
        if step1_5_result.get("details"):
            tprint(f"   📋 Step1_5 Details: {step1_5_result['details']}")

        # Overall assessment
        can_start = step1_passed and step1_5_passed
        if can_start:
            tprint("\n✅ READY TO START FROM STEP2")
            tprint("   Proceeding with existing data...")
        else:
            tprint("\n❌ NOT READY FOR STEP2")
            tprint("   Data validation failed - missing or invalid data")

        tprint("="*80 + "\n")

class ValidationFactory:
    """Factory for creating validators."""

    @staticmethod
    def create_validator(validator_type: str, logger: logging.Logger) -> BaseValidator:
        """Create the appropriate validator."""
        validators = {
            "prerequisites": PrerequisitesValidator,
            "step_validation": StepValidationValidator,
            "data_validation": DataValidationValidator,
        }

        if validator_type not in validators:
            raise ValueError(f"No validator available for: {validator_type}")

        return validators[validator_type](logger)
