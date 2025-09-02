"""
Validator orchestrator for running individual step validators in the training pipeline.
"""

import asyncio
import importlib
import sys
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.prometheus_metrics import metrics
from src.utils.warning_symbols import (
    error,
    missing,
)

class ValidatorOrchestrator:
    """Orchestrator for running step validators in the training pipeline."""

    def __init__(self):
        """Initialize the ValidatorOrchestrator."""
        self.logger = system_logger.getChild("ValidatorOrchestrator")
        self.validators = {}
        self.validation_results = {}
        self.is_initialized = False
        self.pipeline_standards = pipeline_standards
        self.metrics = metrics

    async def initialize(self) -> bool:
        """Initialize ValidatorOrchestrator."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            
            # Load available validators
            await self._load_validators()
            
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    async def _load_validators(self) -> None:
        """Load available step validators dynamically."""
        try:
            # Look for validator modules in the validators directory
            validators_dir = Path(__file__).parent.parent / "validators"
            if validators_dir.exists():
                for validator_file in validators_dir.glob("*_validator.py"):
                    validator_name = validator_file.stem
                    try:
                        module = importlib.import_module(f"src.validators.{validator_name}")
                        if hasattr(module, 'Validator'):
                            self.validators[validator_name] = module.Validator()
                            self.logger.info(f"Loaded validator: {validator_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load validator {validator_name}: {e}")
            
            self.logger.info(f"Loaded {len(self.validators)} validators")
        except Exception as e:
            self.logger.error(f"Error loading validators: {e}")

    async def run_step_validator(
        self,
        step_name: str,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        config: Dict[str, Any],
        validation_level: str = "CRITICAL",
    ) -> Dict[str, Any]:
        """
        Run the validator for a specific step with enhanced validation levels.

        Args:
            step_name: Name of the step (e.g., "step01_data_collection")
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary
            validation_level: Validation level ("BASIC", "STANDARD", "COMPREHENSIVE", "CRITICAL") - defaults to CRITICAL

        Returns:
            Dictionary containing validation results
        """
        start_perf = time.perf_counter()
        
        try:
            self.logger.info(f"🔍 Running {validation_level} validator for {step_name}")

            # Debug-level context for troubleshooting
            self.logger.debug(
                "Input context - training_input_keys=%s pipeline_state_keys=%s validation_level=%s",
                list(training_input.keys()) if isinstance(training_input, dict) else type(training_input).__name__,
                list(pipeline_state.keys()) if isinstance(pipeline_state, dict) else type(pipeline_state).__name__,
                validation_level,
            )

            # Get validator for the step
            validator = await self._get_step_validator(step_name)
            if not validator:
                return self._create_validation_result(
                    step_name, False, f"No validator found for step: {step_name}", validation_level
                )

            # Run validation based on level
            validation_result = await self._run_validation_by_level(
                validator, step_name, training_input, pipeline_state, config, validation_level
            )

            # Record metrics
            validation_time = time.perf_counter() - start_perf
            self.metrics.record_validation_time(step_name, validation_time)
            self.metrics.record_validation_result(step_name, validation_result["is_valid"])

            return validation_result

        except Exception as e:
            self.logger.exception(f"❌ Error running validator for {step_name}: {e}")
            return self._create_validation_result(
                step_name, False, f"Validation error: {str(e)}", validation_level
            )

    async def _get_step_validator(self, step_name: str):
        """Get the appropriate validator for a given step."""
        try:
            # Try to get step-specific validator
            step_key = step_name.replace("step", "").replace("_", "")
            validator_key = f"step{step_key}_validator"
            
            if validator_key in self.validators:
                return self.validators[validator_key]
            
            # Fall back to generic validator
            if "generic_validator" in self.validators:
                return self.validators["generic_validator"]
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting validator for {step_name}: {e}")
            return None

    async def _run_validation_by_level(
        self,
        validator,
        step_name: str,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        config: Dict[str, Any],
        validation_level: str,
    ) -> Dict[str, Any]:
        """Run validation based on the specified level."""
        try:
            if validation_level == "BASIC":
                return await validator.validate_basic(
                    step_name, training_input, pipeline_state, config
                )
            elif validation_level == "STANDARD":
                return await validator.validate_standard(
                    step_name, training_input, pipeline_state, config
                )
            elif validation_level == "COMPREHENSIVE":
                return await validator.validate_comprehensive(
                    step_name, training_input, pipeline_state, config
                )
            elif validation_level == "CRITICAL":
                return await validator.validate_critical(
                    step_name, training_input, pipeline_state, config
                )
            else:
                self.logger.warning(f"Unknown validation level: {validation_level}, using CRITICAL")
                return await validator.validate_critical(
                    step_name, training_input, pipeline_state, config
                )
        except Exception as e:
            self.logger.error(f"Error in validation level {validation_level}: {e}")
            return self._create_validation_result(
                step_name, False, f"Validation level error: {str(e)}", validation_level
            )

    def _create_validation_result(
        self, step_name: str, is_valid: bool, message: str, validation_level: str
    ) -> Dict[str, Any]:
        """Create a standardized validation result."""
        return {
            "step_name": step_name,
            "is_valid": is_valid,
            "message": message,
            "validation_level": validation_level,
            "timestamp": time.time(),
            "details": {}
        }

    async def run_all_step_validators(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        config: Dict[str, Any],
        validation_level: str = "CRITICAL",
    ) -> Dict[str, Any]:
        """
        Run validators for all available steps.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary
            validation_level: Validation level to use for all steps

        Returns:
            Dictionary containing validation results for all steps
        """
        try:
            self.logger.info(f"🚀 Running all step validators with level: {validation_level}")
            
            all_results = {}
            failed_steps = []
            
            # Get list of steps to validate
            steps_to_validate = self._get_steps_to_validate(config)
            
            for step_name in steps_to_validate:
                try:
                    result = await self.run_step_validator(
                        step_name, training_input, pipeline_state, config, validation_level
                    )
                    all_results[step_name] = result
                    
                    if not result["is_valid"]:
                        failed_steps.append(step_name)
                        
                except Exception as e:
                    self.logger.error(f"Error validating step {step_name}: {e}")
                    all_results[step_name] = self._create_validation_result(
                        step_name, False, f"Validation error: {str(e)}", validation_level
                    )
                    failed_steps.append(step_name)

            # Create summary
            summary = {
                "total_steps": len(steps_to_validate),
                "passed_steps": len(steps_to_validate) - len(failed_steps),
                "failed_steps": len(failed_steps),
                "failed_step_names": failed_steps,
                "overall_success": len(failed_steps) == 0,
                "validation_level": validation_level,
                "step_results": all_results
            }

            self.logger.info(f"✅ All step validators completed. {summary['passed_steps']}/{summary['total_steps']} passed")
            
            if failed_steps:
                self.logger.warning(f"⚠️ Failed steps: {failed_steps}")

            return summary

        except Exception as e:
            self.logger.exception(f"❌ Error running all step validators: {e}")
            return {
                "error": str(e),
                "overall_success": False,
                "step_results": {}
            }

    def _get_steps_to_validate(self, config: Dict[str, Any]) -> List[str]:
        """Get the list of steps that should be validated."""
        try:
            # Try to get steps from config
            if "validation_steps" in config:
                return config["validation_steps"]
            
            # Default steps if not specified
            default_steps = [
                "step01_data_collection",
                "step01_5_data_preprocessing",
                "step02_feature_engineering",
                "step03_model_training",
                "step04_model_evaluation"
            ]
            
            return default_steps
        except Exception as e:
            self.logger.error(f"Error getting steps to validate: {e}")
            return []

    async def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        try:
            total_validations = len(self.validation_results)
            successful_validations = sum(1 for r in self.validation_results.values() if r.get("is_valid", False))
            
            return {
                "total_validations": total_validations,
                "successful_validations": successful_validations,
                "failed_validations": total_validations - successful_validations,
                "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
                "last_validation": max(self.validation_results.keys()) if self.validation_results else None,
                "available_validators": list(self.validators.keys())
            }
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {e}")
            return {}

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.logger.info("🧹 Cleaning up ValidatorOrchestrator...")
            
            # Clean up validators
            for validator_name, validator in self.validators.items():
                if hasattr(validator, 'cleanup'):
                    try:
                        await validator.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up validator {validator_name}: {e}")
            
            self.validators.clear()
            self.validation_results.clear()
            self.is_initialized = False
            
            self.logger.info("✅ ValidatorOrchestrator cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Convenience function for creating orchestrator instance
async def create_validator_orchestrator() -> ValidatorOrchestrator:
    """Create and initialize a ValidatorOrchestrator instance."""
    orchestrator = ValidatorOrchestrator()
    success = await orchestrator.initialize()
    if not success:
        raise RuntimeError("Failed to initialize ValidatorOrchestrator")
    return orchestrator
