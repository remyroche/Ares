from ..standardized_parquet_handler import standardized_parquet_handler
"""Step 19: Monte Carlo Validation Validator."""

import asyncio
import json
import os

from typing import Any, Dict, List, Optional

from datetime import datetime

from src.utils.logger import system_logger
from src.training.steps.base_validator import BaseValidator
from src.utils.warning_symbols import success, error, failed

logger = system_logger

class Step19MonteCarloValidationValidator(BaseValidator):
    """Validator for Step 19 Monte Carlo Validation outputs."""

    def __init__(self):
        super().__init__("step19_monte_carlo_validation")
        self.required_files = [
            "monte_carlo_results.json",
            "monte_carlo_performance.json",
            "monte_carlo_metadata.json"
        ]

    async def validate_step(self, symbol: str, exchange: str, data_dir: str = "data/training") -> Dict[str, Any]:
        """Validate Monte Carlo validation step outputs."""
        try:
            self.logger.info(f"🔍 Validating Step 19 Monte Carlo Validation for {exchange} {symbol}")

            validation_results = {
                "step_name": "step19_monte_carlo_validation",
                "symbol": symbol,
                "exchange": exchange,
                "validation_time": datetime.now().isoformat(),
                "checks": {},
                "passed": True,
                "critical_failures": []
            }

            # Check required files exist
            files_exist = await self._validate_required_files(data_dir, symbol, exchange)
            validation_results["checks"]["required_files"] = files_exist

            if not files_exist["passed"]:
                validation_results["passed"] = False
                validation_results["critical_failures"].append("Missing required output files")
                return validation_results

            # Load and validate results
            results_data = await self._load_results_data(data_dir, symbol, exchange)
            if results_data is None:
                validation_results["passed"] = False
                validation_results["critical_failures"].append("Could not load results data")
                return validation_results

            # Validate Monte Carlo results structure
            mc_results_valid = self._validate_monte_carlo_results(results_data.get("results", {}))
            validation_results["checks"]["monte_carlo_results"] = mc_results_valid

            # Validate performance metrics
            performance_valid = self._validate_performance_metrics(results_data.get("performance", {}))
            validation_results["checks"]["performance_metrics"] = performance_valid

            # Validate simulation metadata
            metadata_valid = self._validate_simulation_metadata(results_data.get("metadata", {}))
            validation_results["checks"]["simulation_metadata"] = metadata_valid

            # Validate statistical significance
            statistical_valid = self._validate_statistical_significance(results_data.get("results", {}))
            validation_results["checks"]["statistical_significance"] = statistical_valid

            # Validate risk metrics
            risk_valid = self._validate_risk_metrics(results_data.get("performance", {}))
            validation_results["checks"]["risk_metrics"] = risk_valid

            # Overall validation result
            all_checks_passed = all([
                mc_results_valid["passed"],
                performance_valid["passed"],
                metadata_valid["passed"],
                statistical_valid["passed"],
                risk_valid["passed"]
            ])

            validation_results["passed"] = all_checks_passed

            if not all_checks_passed:
                validation_results["critical_failures"].append("One or more validation checks failed")

            self.logger.info(f"✅ Step 19 validation completed: {'PASSED' if validation_results['passed'] else 'FAILED'}")
            return validation_results

        except Exception as e:
            self.logger.error(f"❌ Error validating Step 19: {e}")
            return {
                "step_name": "step19_monte_carlo_validation",
                "passed": False,
                "error": str(e),
                "validation_time": datetime.now().isoformat()
            }

    async def _validate_required_files(self, data_dir: str, symbol: str, exchange: str) -> Dict[str, Any]:
        """Validate that all required output files exist."""
        result = {"passed": True, "missing_files": [], "existing_files": []}

        for filename in self.required_files:
            filepath = os.path.join(data_dir, f"{exchange}_{symbol}_{filename}")
            if os.path.exists(filepath):
                result["existing_files"].append(filename)
            else:
                result["missing_files"].append(filename)
                result["passed"] = False

        return result

    async def _load_results_data(self, data_dir: str, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """Load Monte Carlo validation results from JSON files."""
        try:
            results = {}

            # Load results file
            results_file = os.path.join(data_dir, f"{exchange}_{symbol}_monte_carlo_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results["results"] = json.load(f)

            # Load performance file
            perf_file = os.path.join(data_dir, f"{exchange}_{symbol}_monte_carlo_performance.json")
            if os.path.exists(perf_file):
                with open(perf_file, 'r') as f:
                    results["performance"] = json.load(f)

            # Load metadata file
            meta_file = os.path.join(data_dir, f"{exchange}_{symbol}_monte_carlo_metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    results["metadata"] = json.load(f)

            return results if results else None

        except Exception as e:
            self.logger.error(f"Error loading results data: {e}")
            return None

    def _validate_monte_carlo_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Monte Carlo results structure and content."""
        result = {"passed": True, "issues": []}

        required_fields = [
            "symbol", "exchange", "validation_date", "validation_method",
            "simulation_count", "p_value", "confidence_intervals", "effect_size"
        ]

        for field in required_fields:
            if field not in results:
                result["issues"].append(f"Missing required field: {field}")
                result["passed"] = False

        # Validate simulation count
        if "simulation_count" in results:
            sim_count = results["simulation_count"]
            if not isinstance(sim_count, int) or sim_count <= 0:
                result["issues"].append("Invalid simulation count")
                result["passed"] = False
            elif sim_count < 100:
                result["issues"].append("Simulation count too low for reliable results")
                result["passed"] = False

        # Validate p-value
        if "p_value" in results:
            p_val = results["p_value"]
            if not isinstance(p_val, (int, float)) or not (0 <= p_val <= 1):
                result["issues"].append("Invalid p-value")
                result["passed"] = False

        # Validate confidence intervals
        if "confidence_intervals" in results:
            ci = results["confidence_intervals"]
            if not isinstance(ci, dict) or "95_percent_ci" not in ci:
                result["issues"].append("Invalid confidence intervals structure")
                result["passed"] = False

        return result

    def _validate_performance_metrics(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance metrics structure and reasonableness."""
        result = {"passed": True, "issues": []}

        # Check distribution statistics
        if "distribution_stats" not in performance:
            result["issues"].append("Missing distribution statistics")
            result["passed"] = False
        else:
            dist_stats = performance["distribution_stats"]
            required_stats = ["mean_return", "std_return", "mean_sharpe"]

            for stat in required_stats:
                if stat not in dist_stats:
                    result["issues"].append(f"Missing distribution stat: {stat}")
                    result["passed"] = False

            # Validate Sharpe ratio reasonableness
            if "mean_sharpe" in dist_stats:
                sharpe = dist_stats["mean_sharpe"]
                if not isinstance(sharpe, (int, float)) or abs(sharpe) > 10:
                    result["issues"].append("Unreasonable Sharpe ratio value")
                    result["passed"] = False

        # Check percentiles
        if "percentiles" not in performance:
            result["issues"].append("Missing percentile data")
            result["passed"] = False

        # Check risk metrics
        if "risk_metrics" not in performance:
            result["issues"].append("Missing risk metrics")
            result["passed"] = False
        else:
            risk_metrics = performance["risk_metrics"]
            if "var_95_mean" not in risk_metrics:
                result["issues"].append("Missing VaR 95% metric")
                result["passed"] = False

        return result

    def _validate_simulation_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate simulation metadata."""
        result = {"passed": True, "issues": []}

        # Check simulation parameters
        if "simulation_parameters" not in metadata:
            result["issues"].append("Missing simulation parameters")
            result["passed"] = False
        else:
            sim_params = metadata["simulation_parameters"]
            required_params = ["random_seed", "sample_size", "bootstrap_method"]

            for param in required_params:
                if param not in sim_params:
                    result["issues"].append(f"Missing simulation parameter: {param}")
                    result["passed"] = False

        # Check convergence metrics
        if "convergence_metrics" not in metadata:
            result["issues"].append("Missing convergence metrics")
            result["passed"] = False
        else:
            conv_metrics = metadata["convergence_metrics"]
            if "converged" not in conv_metrics:
                result["issues"].append("Missing convergence status")
                result["passed"] = False

        # Check robustness metrics
        if "robustness_metrics" not in metadata:
            result["issues"].append("Missing robustness metrics")
            result["passed"] = False

        return result

    def _validate_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical significance of results."""
        result = {"passed": True, "issues": []}

        # Check for statistical significance
        if "p_value" in results:
            p_val = results["p_value"]
            if p_val > 0.1:  # Very liberal threshold for Monte Carlo
                result["issues"].append(f"Results not statistically significant (p={p_val})")
                result["passed"] = False

        # Check effect size
        if "effect_size" in results:
            effect_size = results["effect_size"]
            if abs(effect_size) < 0.1:  # Small effect size
                result["issues"].append("Effect size is very small")
                # Not a critical failure, just a warning

        # Check confidence intervals
        if "confidence_intervals" in results:
            ci_95 = results["confidence_intervals"].get("95_percent_ci", [])
            if len(ci_95) == 2:
                ci_width = abs(ci_95[1] - ci_95[0])
                if ci_width > 1.0:  # Very wide confidence interval
                    result["issues"].append("Confidence interval is very wide")
                    # Not a critical failure

        return result

    def _validate_risk_metrics(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk metrics are reasonable."""
        result = {"passed": True, "issues": []}

        if "risk_metrics" in performance:
            risk = performance["risk_metrics"]

            # Validate VaR is negative (loss)
            if "var_95_mean" in risk:
                var_95 = risk["var_95_mean"]
                if var_95 > 0:
                    result["issues"].append("VaR should be negative (representing loss)")
                    result["passed"] = False

            # Validate VaR magnitude is reasonable
            if "var_95_worst" in risk:
                var_worst = risk["var_95_worst"]
                if abs(var_worst) > 0.5:  # 50% loss in worst case
                    result["issues"].append("Extreme VaR value detected")
                    result["passed"] = False

        # Check stability metrics
        if "stability_metrics" in performance:
            stability = performance["stability_metrics"]

            if "coefficient_of_variation" in stability:
                cv = stability["coefficient_of_variation"]
                if cv > 5:  # Extremely high variability
                    result["issues"].append("Results show extreme variability")
                    result["passed"] = False

        return result

# For backward compatibility
async def validate_step19_monte_carlo_validation(symbol: str, exchange: str, data_dir: str = "data/training") -> Dict[str, Any]:
    """Validate Step 19 Monte Carlo Validation."""
    validator = Step19MonteCarloValidationValidator()
    return await validator.validate_step(symbol, exchange, data_dir)

if __name__ == "__main__":
    # Test the validator
    async def test():
        result = await validate_step19_monte_carlo_validation("ETHUSDT", "BINANCE")
        print(f"Validation result: {result}")

    asyncio.run(test())
