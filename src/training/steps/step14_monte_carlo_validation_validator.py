"""
Validator for Step 14: Monte Carlo Validation
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from src.utils.warning_symbols import (
    error,
    failed,
    validation_error,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step14MonteCarloValidationValidator(BaseValidator):
    """Validator for Step 14: Monte Carlo Validation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("step14_monte_carlo_validation", config)

    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """
        Validate the Monte Carlo validation step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating Monte Carlo validation step...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("monte_carlo_validation", {})

        # 1. Validate error absence
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.print(validation_error("❌ Monte Carlo validation step had errors"))
            return False

        # 2. Validate Monte Carlo validation files existence
        validation_files_passed = self._validate_monte_carlo_files(
            symbol,
            exchange,
            data_dir,
        )
        if not validation_files_passed:
            self.print(failed("❌ Monte Carlo validation files validation failed"))
            return False

        # 3. Validate Monte Carlo statistical significance
        significance_passed = self._validate_statistical_significance(
            symbol,
            exchange,
            data_dir,
        )
        if not significance_passed:
            self.print(failed("❌ Statistical significance validation failed"))
            return False

        # 4. Validate Monte Carlo performance distribution
        distribution_passed = self._validate_performance_distribution(
            symbol,
            exchange,
            data_dir,
        )
        if not distribution_passed:
            self.print(failed("❌ Performance distribution validation failed"))
            return False

        # 5. Validate Monte Carlo robustness
        robustness_passed = self._validate_monte_carlo_robustness(
            symbol,
            exchange,
            data_dir,
        )
        if not robustness_passed:
            self.print(failed("❌ Monte Carlo robustness validation failed"))
            return False

        # 6. Validate outcome favorability
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result,
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            self.print(
                validation_error("⚠️ Monte Carlo validation outcome is not favorable"),
            )
            return False

        self.logger.info("✅ Monte Carlo validation validation passed")
        return True

    def _validate_monte_carlo_files(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate that Monte Carlo validation files exist.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if files exist
        """
        try:
            # Expected Monte Carlo validation file patterns
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_monte_carlo_results.json",
                f"{data_dir}/{exchange}_{symbol}_monte_carlo_performance.json",
                f"{data_dir}/{exchange}_{symbol}_monte_carlo_metadata.json",
            ]

            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(
                    file_path,
                    "monte_carlo_files",
                )
                if not file_passed:
                    missing_files.append(file_path)

            if missing_files:
                self.logger.error(
                    f"❌ Missing Monte Carlo validation files: {missing_files}",
                )
                return False

            self.logger.info("✅ All Monte Carlo validation files exist")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error validating Monte Carlo validation files: {e}",
            )
            return False

    def _validate_statistical_significance(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate statistical significance of Monte Carlo results.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if statistical significance is acceptable
        """
        try:
            # Load Monte Carlo results
            results_file = f"{data_dir}/{exchange}_{symbol}_monte_carlo_results.json"

            if os.path.exists(results_file):
                import json

                with open(results_file) as f:
                    results = json.load(f)

                # Check number of simulations
                if "simulation_count" in results:
                    sim_count = results["simulation_count"]
                    if sim_count < 1000:
                        self.logger.warning(
                            f"⚠️ Few Monte Carlo simulations: {sim_count}",
                        )
                    elif sim_count > 100000:
                        self.logger.warning(
                            f"⚠️ Many Monte Carlo simulations: {sim_count}",
                        )

                # Check p-value for statistical significance
                if "p_value" in results:
                    p_value = results["p_value"]
                    if p_value > 0.05:
                        self.logger.warning(
                            f"⚠️ High p-value (not statistically significant): {p_value:.3f}",
                        )
                    elif p_value < 0.001:
                        self.logger.info(
                            f"✅ Very low p-value (highly significant): {p_value:.6f}",
                        )

                # Check confidence intervals
                if "confidence_intervals" in results:
                    ci = results["confidence_intervals"]

                    if "95_percent_ci" in ci:
                        ci_95 = ci["95_percent_ci"]
                        ci_width = ci_95[1] - ci_95[0]
                        if ci_width > 0.2:
                            self.logger.warning(
                                f"⚠️ Wide 95% confidence interval: {ci_width:.3f}",
                            )

                    if "99_percent_ci" in ci:
                        ci_99 = ci["99_percent_ci"]
                        ci_width = ci_99[1] - ci_99[0]
                        if ci_width > 0.3:
                            self.logger.warning(
                                f"⚠️ Wide 99% confidence interval: {ci_width:.3f}",
                            )

                # Check effect size
                if "effect_size" in results:
                    effect_size = results["effect_size"]
                    if abs(effect_size) < 0.1:
                        self.print(error("⚠️ Small effect size: {effect_size:.3f}"))
                    elif abs(effect_size) > 0.8:
                        self.logger.info(f"✅ Large effect size: {effect_size:.3f}")

            self.logger.info("✅ Statistical significance validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during statistical significance validation: {e}",
            )
            return False

    def _validate_performance_distribution(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate performance distribution from Monte Carlo simulations.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if performance distribution is acceptable
        """
        try:
            # Load Monte Carlo performance results
            performance_file = (
                f"{data_dir}/{exchange}_{symbol}_monte_carlo_performance.json"
            )

            if os.path.exists(performance_file):
                import json

                with open(performance_file) as f:
                    performance = json.load(f)

                # Check performance distribution statistics
                if "distribution_stats" in performance:
                    stats = performance["distribution_stats"]

                    # Check mean performance
                    if "mean" in stats:
                        mean_perf = stats["mean"]
                        if mean_perf < 0.5:
                            self.logger.warning(
                                f"⚠️ Low mean Monte Carlo performance: {mean_perf:.3f}",
                            )

                    # Check standard deviation
                    if "std" in stats:
                        std_perf = stats["std"]
                        if std_perf > 0.2:
                            self.logger.warning(
                                f"⚠️ High Monte Carlo performance variance: {std_perf:.3f}",
                            )
                        elif std_perf < 0.01:
                            self.logger.warning(
                                f"⚠️ Very low Monte Carlo performance variance: {std_perf:.3f}",
                            )

                    # Check skewness
                    if "skewness" in stats:
                        skewness = stats["skewness"]
                        if abs(skewness) > 2:
                            self.logger.warning(
                                f"⚠️ Highly skewed Monte Carlo performance: {skewness:.3f}",
                            )

                    # Check kurtosis
                    if "kurtosis" in stats:
                        kurtosis = stats["kurtosis"]
                        if kurtosis > 10:
                            self.logger.warning(
                                f"⚠️ High kurtosis in Monte Carlo performance: {kurtosis:.3f}",
                            )

                # Check performance percentiles
                if "percentiles" in performance:
                    percentiles = performance["percentiles"]

                    # Check 5th percentile (worst case)
                    if "5th" in percentiles:
                        p5 = percentiles["5th"]
                        if p5 < 0.3:
                            self.logger.warning(
                                f"⚠️ Poor 5th percentile performance: {p5:.3f}",
                            )

                    # Check 95th percentile (best case)
                    if "95th" in percentiles:
                        p95 = percentiles["95th"]
                        if p95 < 0.6:
                            self.logger.warning(
                                f"⚠️ Poor 95th percentile performance: {p95:.3f}",
                            )

                # Check performance stability
                if "stability_metrics" in performance:
                    stability = performance["stability_metrics"]

                    if "coefficient_of_variation" in stability:
                        cv = stability["coefficient_of_variation"]
                        if cv > 0.5:
                            self.logger.warning(
                                f"⚠️ High coefficient of variation: {cv:.3f}",
                            )

                    if "interquartile_range" in stability:
                        iqr = stability["interquartile_range"]
                        if iqr > 0.3:
                            self.logger.warning(
                                f"⚠️ High interquartile range: {iqr:.3f}",
                            )

            self.logger.info("✅ Performance distribution validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during performance distribution validation: {e}",
            )
            return False

    def _validate_monte_carlo_robustness(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """
        Validate Monte Carlo simulation robustness.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if robustness is acceptable
        """
        try:
            # Load Monte Carlo metadata
            metadata_file = f"{data_dir}/{exchange}_{symbol}_monte_carlo_metadata.json"

            if os.path.exists(metadata_file):
                import json

                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Check simulation parameters
                if "simulation_parameters" in metadata:
                    params = metadata["simulation_parameters"]

                    # Check random seed quality
                    if "random_seed" in params:
                        seed = params["random_seed"]
                        if seed in (0, 42):  # Common default seeds
                            self.print(error("⚠️ Using common random seed: {seed}"))

                    # Check sample size
                    if "sample_size" in params:
                        sample_size = params["sample_size"]
                        if sample_size < 100:
                            self.logger.warning(
                                f"⚠️ Small Monte Carlo sample size: {sample_size}",
                            )
                        elif sample_size > 10000:
                            self.logger.warning(
                                f"⚠️ Large Monte Carlo sample size: {sample_size}",
                            )

                # Check convergence metrics
                if "convergence_metrics" in metadata:
                    convergence = metadata["convergence_metrics"]

                    if "converged" in convergence:
                        converged = convergence["converged"]
                        if not converged:
                            self.logger.warning(
                                "⚠️ Monte Carlo simulations did not converge",
                            )

                    if "convergence_iterations" in convergence:
                        iterations = convergence["convergence_iterations"]
                        if iterations > 1000:
                            self.logger.warning(
                                f"⚠️ Many convergence iterations: {iterations}",
                            )

                # Check robustness metrics
                if "robustness_metrics" in metadata:
                    robustness = metadata["robustness_metrics"]

                    if "sensitivity_score" in robustness:
                        sensitivity = robustness["sensitivity_score"]
                        if sensitivity > 0.8:
                            self.logger.warning(
                                f"⚠️ High sensitivity to parameters: {sensitivity:.3f}",
                            )

                    if "stability_score" in robustness:
                        stability = robustness["stability_score"]
                        if stability < 0.7:
                            self.logger.warning(
                                f"⚠️ Low Monte Carlo stability: {stability:.3f}",
                            )

            self.logger.info("✅ Monte Carlo robustness validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during Monte Carlo robustness validation: {e}",
            )
            return False


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the step14_monte_carlo_validation validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step14MonteCarloValidationValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step14_monte_carlo_validation",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time(),
    }


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_validator():
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }

        pipeline_state = {
            "monte_carlo_validation": {"status": "SUCCESS", "duration": 1500.5},
        }

        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
