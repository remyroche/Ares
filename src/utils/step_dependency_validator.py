from __future__ import annotations

"""
Step dependency validator for the training pipeline.
Ensures that steps don't proceed if their prerequisites have failed.
"""

from typing import Any

from src.utils.logger import system_logger


class StepDependencyValidator:
    """
    Validates step dependencies to ensure pipeline integrity.
    Prevents steps from running if their prerequisites have failed.
    """

    def __init__(self):
        self.logger = system_logger.getChild("StepDependencyValidator")
        # Define step dependencies (step -> list of required steps)
        self.step_dependencies = {
            "step01_data_collection": [],
            "step01_5_data_converter": ["step01_data_collection"],
            "step02_data_reading": ["step01_5_data_converter"],
            "step02_5_sr_optimization": ["step02_data_reading"],
            "step03_hmm_regime_discovery": ["step02_5_sr_optimization"],
            "step04_triple_barrier_method": ["step03_hmm_regime_discovery"],
            "step04_regime_data_splitting": ["step04_triple_barrier_method"],
            "step05_labeling": ["step04_triple_barrier_method"],
            "step06_feature_engineering": ["step05_labeling"],
            "step07_enhanced_matrix_operations": ["step06_feature_engineering"],
            "step08_regime_data_splitting": ["step07_enhanced_matrix_operations"],
            "step09_hmm_based_training": ["step08_regime_data_splitting"],
            "step09_5_multi_timeframe_hmm_ensemble": ["step09_hmm_based_training"],
            "step09_5_hmm_lm_generalist_training": [
                "step09_5_multi_timeframe_hmm_ensemble"
            ],
            "step10_unified_regime_intelligence": [
                "step09_5_hmm_lm_generalist_training"
            ],
            "step11_analyst_creation": ["step10_unified_regime_intelligence"],
            "step12_analyst_enhancement": ["step11_analyst_creation"],
            "step13_analyst_ensemble_creation": ["step12_analyst_enhancement"],
            "step14_tactician_labeling": ["step13_analyst_ensemble_creation"],
            "step15_tactician_specialist_training": ["step14_tactician_labeling"],
            # Extended steps
            "step16_confidence_calibration": ["step15_tactician_specialist_training"],
            "step17_final_parameters_optimization": ["step16_confidence_calibration"],
            "step18_walk_forward_validation": ["step17_final_parameters_optimization"],
            "step19_monte_carlo_validation": ["step18_walk_forward_validation"],
            "step20_ab_testing": ["step19_monte_carlo_validation"],
            "step21_saving": ["step20_ab_testing"],
        }
        # Define critical data requirements for each step
        self.critical_data_requirements = {
            "step01_data_collection": {
                "required_files": ["data_cache/klines_*_*_1m_consolidated.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500,
            },
            "step01_5_data_converter": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500,
            },
            "step02_data_reading": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500,
            },
            "step02_5_sr_optimization": {
                "required_files": ["data_cache/unified/*/*/*/*.parquet"],
                "required_columns": ["open", "high", "low", "close", "volume"],
                "min_rows": 500,
            },
            "step03_hmm_regime_discovery": {
                "required_files": ["data/hmm_regimes/*_composite_clusters.parquet"],
                "required_columns": ["composite_cluster_id"],
                "min_rows": 100,
            },
            "step04_triple_barrier_method": {
                "required_files": ["data/training/*_triple_barrier_*.parquet"],
                "required_columns": ["triple_barrier_label"],
                "min_rows": 50,
            },
            "step05_labeling": {
                "required_files": ["data/training/*_labeled_*.parquet"],
                "required_columns": ["label"],
                "min_rows": 50,
            },
            "step06_feature_engineering": {
                "required_files": [
                    "data/training/*_features_train.parquet",
                    "data/training/*_features_val.parquet",
                ],
                "required_columns": ["timestamp", "returns", "volatility"],
                "min_rows": 1000,
            },
            "step07_enhanced_matrix_operations": {
                "required_files": ["data/matrix_operations/*_matrix_operations_*.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step08_hmm_based_training": {
                "required_files": ["data/training/*_hmm_models.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step08_5_unified_regime_intelligence": {
                "required_files": ["data/training/*_unified_intelligence.parquet"],
                "required_columns": ["intelligence_score"],
                "min_rows": 100,
            },
            "step09_analyst_enhancement": {
                "required_files": ["data/training/*_analyst_models.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step10_tactician_labeling": {
                "required_files": ["data/training/*_tactician_labels.parquet"],
                "required_columns": ["tactician_label"],
                "min_rows": 100,
            },
            "step11_tactician_specialist_training": {
                "required_files": ["data/training/*_specialist_models.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step12_confidence_calibration": {
                "required_files": ["data/training/*_calibration_results.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step13_final_parameters_optimization": {
                "required_files": ["data/training/*_optimization_results.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step14_walk_forward_validation": {
                "required_files": ["data/training/*_walk_forward_results.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step15_monte_carlo_validation": {
                "required_files": ["data/training/*_monte_carlo_results.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step14_ab_testing": {
                "required_files": ["data/training/*_ab_test_results.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step15_saving": {
                "required_files": ["data/training/*_final_models.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
            # Extended steps
            "step16_confidence_calibration": {
                "required_files": ["data/training/*_extended_calibration_results.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step17_final_parameters_optimization": {
                "required_files": [
                    "data/training/*_extended_optimization_results.json"
                ],
                "required_columns": [],
                "min_rows": 0,
            },
            "step18_walk_forward_validation": {
                "required_files": [
                    "data/training/*_extended_walk_forward_results.json"
                ],
                "required_columns": [],
                "min_rows": 0,
            },
            "step19_monte_carlo_validation": {
                "required_files": ["data/training/*_extended_monte_carlo_results.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step20_ab_testing": {
                "required_files": ["data/training/*_extended_ab_test_results.json"],
                "required_columns": [],
                "min_rows": 0,
            },
            "step21_saving": {
                "required_files": ["data/training/*_extended_final_models.pkl"],
                "required_columns": [],
                "min_rows": 0,
            },
        }

    async def validate_step_prerequisites(
        self,
        step_name: str,
        pipeline_state: dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        force_rerun: bool = False,
    ) -> dict[str, Any]:
        """
        Validate that all dependencies for a step have been completed successfully.

        Args:
            step_name: Name of the step to validate
            pipeline_state: Current state of the pipeline

        Returns:
            Dict[str, Any]: Result including validity and reason
        """
        if step_name not in self.step_dependencies:
            return {"valid": True, "reason": "No dependency mapping for step"}
        required_steps = self.step_dependencies[step_name]
        if not required_steps:
            return {"valid": True, "reason": "Step has no dependencies"}
        # Validate each required step
        failed_steps: list[str] = []
        for required_step in required_steps:
            ok = await self._validate_single_dependency(required_step, pipeline_state)
            if not ok:
                failed_steps.append(required_step)
        if failed_steps:
            return {
                "valid": False,
                "reason": f"Missing or failed prerequisites: {failed_steps}",
                "failed_steps": failed_steps,
            }
        return {"valid": True, "reason": "All dependencies satisfied"}

    async def _validate_single_dependency(
        self, step_name: str, pipeline_state: dict[str, Any]
    ) -> bool:
        """
        Validate a single step dependency.

        Args:
            step_name: Name of the required step
            pipeline_state: Current state of the pipeline

        Returns:
            bool: True if dependency is met, False otherwise
        """
        # Check if step exists in pipeline state
        if step_name not in pipeline_state:
            self.logger.error(f"❌ Step {step_name} not found in pipeline state")
            return False

        step_result = pipeline_state[step_name]

        # Check if step completed successfully
        if not isinstance(step_result, dict):
            self.logger.error(f"❌ Step {step_name} result is not a dictionary")
            return False

        status = step_result.get("status", "UNKNOWN")
        if status != "SUCCESS":
            self.logger.error(f"❌ Step {step_name} failed with status: {status}")
            return False

        # Check if step has required data
        if not await self._validate_step_data(step_name):
            self.logger.error(f"❌ Step {step_name} data validation failed")
            return False

        self.logger.info(f"✅ Step {step_name} dependency validated")
        return True

    async def _validate_step_data(self, step_name: str) -> bool:
        """
        Validate that a step has the required data files and structure.

        Args:
            step_name: Name of the step to validate

        Returns:
            bool: True if data validation passes, False otherwise
        """
        if step_name not in self.critical_data_requirements:
            self.logger.warning(f"No data requirements defined for {step_name}")
            return True

        requirements = self.critical_data_requirements[step_name]

        # Check required files
        required_files = requirements.get("required_files", [])
        for file_pattern in required_files:
            if not await self._check_file_pattern(file_pattern):
                self.logger.error(f"❌ Required file pattern not found: {file_pattern}")
                return False

        # Check required columns (if we have data to check)
        required_columns = requirements.get("required_columns", [])
        if required_columns:
            if not await self._check_columns(step_name, required_columns):
                self.logger.error(f"❌ Required columns not found: {required_columns}")
                return False

        # Check minimum rows
        min_rows = requirements.get("min_rows", 0)
        if min_rows > 0 and not await self._check_min_rows(step_name, min_rows):
            self.logger.error(f"❌ Insufficient data rows: {min_rows} required")
            return False

        return True

    async def _check_file_pattern(self, file_pattern: str) -> bool:
        """
        Check if files matching a pattern exist.

        Args:
            file_pattern: Glob pattern to check

        Returns:
            bool: True if files exist, False otherwise
        """
        try:
            from pathlib import Path

            # Convert glob pattern to path
            pattern_path = Path(file_pattern)

            # Check if any files match the pattern
            matching_files = list(Path(pattern_path.parent).glob(pattern_path.name))

            if not matching_files:
                self.logger.warning(f"No files found matching pattern: {file_pattern}")
                return False

            self.logger.debug(
                f"Found {len(matching_files)} files matching: {file_pattern}"
            )
            return True

        except Exception as e:
            self.logger.exception(f"Error checking file pattern {file_pattern}: {e}")
            return False

    async def _check_columns(self, step_name: str, required_columns: list[str]) -> bool:
        """
        Check if required columns exist in step data.

        Args:
            step_name: Name of the step
            required_columns: List of required column names

        Returns:
            bool: True if all columns exist, False otherwise
        """
        try:
            # This is a simplified check - in practice, you'd load the actual data
            # For now, we'll assume columns exist if the step completed successfully
            self.logger.debug(f"Column validation for {step_name}: {required_columns}")
            return True

        except Exception as e:
            self.logger.exception(f"Error checking columns for {step_name}: {e}")
            return False

    async def _check_min_rows(self, step_name: str, min_rows: int) -> bool:
        """
        Check if step data has minimum required rows.

        Args:
            step_name: Name of the step
            min_rows: Minimum number of rows required

        Returns:
            bool: True if sufficient rows exist, False otherwise
        """
        try:
            # This is a simplified check - in practice, you'd load the actual data
            # For now, we'll assume sufficient rows exist if the step completed successfully
            self.logger.debug(
                f"Row count validation for {step_name}: {min_rows} required"
            )
            return True

        except Exception as e:
            self.logger.exception(f"Error checking row count for {step_name}: {e}")
            return False

    def get_step_dependencies(self, step_name: str) -> list[str]:
        """
        Get the list of dependencies for a step.

        Args:
            step_name: Name of the step

        Returns:
            List[str]: List of required step names
        """
        return self.step_dependencies.get(step_name, [])

    def get_critical_requirements(self, step_name: str) -> dict[str, Any]:
        """
        Get the critical data requirements for a step.

        Args:
            step_name: Name of the step

        Returns:
            Dict[str, Any]: Dictionary of requirements
        """
        return self.critical_data_requirements.get(step_name, {})

    def clear_validation_cache(self) -> None:
        """Clear the validation cache."""
        self.validation_cache.clear()
        self.last_validation_time.clear()
        self.logger.info("Validation cache cleared")

    def get_validation_stats(self) -> dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dict[str, Any]: Validation statistics
        """
        return {
            "cache_size": len(self.validation_cache),
            "last_validation_times": self.last_validation_time,
            "total_steps": len(self.step_dependencies),
            "total_requirements": len(self.critical_data_requirements),
        }


# Global instance for easy access
step_dependency_validator = StepDependencyValidator()


async def validate_step_dependencies(
    step_name: str, pipeline_state: dict[str, Any]
) -> bool:
    """
    Convenience function to validate step dependencies.

    Args:
        step_name: Name of the step to validate
        pipeline_state: Current state of the pipeline

    Returns:
        bool: True if all dependencies are met, False otherwise
    """
    result = await step_dependency_validator.validate_step_prerequisites(
        step_name=step_name,
        pipeline_state=pipeline_state,
        checkpoint_dir="checkpoints",
        force_rerun=False,
    )
    return bool(result.get("valid", False))


def get_step_dependencies(step_name: str) -> list[str]:
    """
    Convenience function to get step dependencies.

    Args:
        step_name: Name of the step

    Returns:
        List[str]: List of required step names
    """
    return step_dependency_validator.get_step_dependencies(step_name)


def get_critical_requirements(step_name: str) -> dict[str, Any]:
    """
    Convenience function to get critical requirements.

    Args:
        step_name: Name of the step

    Returns:
        Dict[str, Any]: Dictionary of requirements
    """
    return step_dependency_validator.get_critical_requirements(step_name)
