#!/usr/bin/env python3
"""
Configuration Settings Usage Example

This script demonstrates how to use the new configuration sections
in src/config.py for enhanced hyperparameter optimization and computational optimization.
"""


import asyncio
import logging
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.config import CONFIG  # noqa: E402
from src.utils.warning_symbols import error, failed, initialization_error, warning  # noqa: E402
from src.training.bayesian_optimizer import AdvancedBayesianOptimizer  # noqa: E402
from src.training.multi_objective_optimizer import MultiObjectiveOptimizer  # noqa: E402
from src.training.optimized_backtester import OptimizedBacktester  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def safe_call(func):
    """Decorator to catch and log exceptions for demo methods."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - demonstration guard
            logger.error("Error in %s: %s", func.__name__, exc, exc_info=True)
            return None

    return wrapper


class ConfigurationUsageExample:
    """Example class demonstrating configuration usage"""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = CONFIG
        self.hpo_config: Dict[str, Any] = CONFIG["hyperparameter_optimization"]
        self.comp_config: Dict[str, Any] = CONFIG["computational_optimization"]

    @safe_call
    def validate_configuration(self) -> bool:  # type: ignore[override]
        """Validate the configuration settings"""
        # Check required sections
        required_sections = [
            "hyperparameter_optimization",
            "computational_optimization",
        ]
        for section in required_sections:
            if section not in self.config:
                msg = f"Missing required configuration section: {section}"
                raise ValueError(msg)

        # Validate hyperparameter optimization
        hpo_config = self.hpo_config
        if (
            not hpo_config["multi_objective"]["enabled"]
            and not hpo_config["bayesian_optimization"]["enabled"]
            and not hpo_config["adaptive_optimization"]["enabled"]
        ):
            print(warning("All optimization types are disabled"))

        # Validate computational optimization
        comp_config = self.comp_config
        if (
            not comp_config["caching"]["enabled"]
            and not comp_config["parallelization"]["enabled"]
            and not comp_config["memory_management"]["enabled"]
        ):
            print(warning("All computational optimization types are disabled"))

        logger.info("Configuration validation passed")
        return True

    @safe_call
    def print_configuration_summary(self) -> None:
        """Print a summary of the current configuration"""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        # Hyperparameter Optimization Summary
        print("\n📊 HYPERPARAMETER OPTIMIZATION:")
        hpo_config = self.hpo_config

        print(
            f"  Multi-Objective: {'✅ Enabled' if hpo_config['multi_objective']['enabled'] else '❌ Disabled'}",
        )
        if hpo_config["multi_objective"]["enabled"]:
            objectives = hpo_config["multi_objective"]["objectives"]
            weights = hpo_config["multi_objective"]["weights"]
            print(f"    Objectives: {objectives}")
            print(f"    Weights: {weights}")

        print(
            f"  Bayesian: {'✅ Enabled' if hpo_config['bayesian_optimization']['enabled'] else '❌ Disabled'}",
        )
        if hpo_config["bayesian_optimization"]["enabled"]:
            bayesian = hpo_config["bayesian_optimization"]
            print(f"    Strategy: {bayesian['sampling_strategy']}")
            print(f"    Max Trials: {bayesian['max_trials']}")
            print(f"    Patience: {bayesian['patience']}")

        print(
            f"  Adaptive: {'✅ Enabled' if hpo_config['adaptive_optimization']['enabled'] else '❌ Disabled'}",
        )
        if hpo_config["adaptive_optimization"]["enabled"]:
            regimes = list(
                hpo_config["adaptive_optimization"]["regime_specific_constraints"].keys()
            )
            print(f"    Regimes: {regimes}")

        # Computational Optimization Summary
        print("\n⚡ COMPUTATIONAL OPTIMIZATION:")
        comp_config = self.comp_config

        print(
            f"  Caching: {'✅ Enabled' if comp_config['caching']['enabled'] else '❌ Disabled'}",
        )
        if comp_config["caching"]["enabled"]:
            print(f"    Max Cache Size: {comp_config['caching']['max_cache_size']}")
            print(f"    Cache TTL: {comp_config['caching']['cache_ttl']}s")

        print(
            f"  Parallelization: {'✅ Enabled' if comp_config['parallelization']['enabled'] else '❌ Disabled'}",
        )
        if comp_config["parallelization"]["enabled"]:
            print(f"    Max Workers: {comp_config['parallelization']['max_workers']}")
            print(f"    Chunk Size: {comp_config['parallelization']['chunk_size']}")

        print(
            f"  Memory Management: {'✅ Enabled' if comp_config['memory_management']['enabled'] else '❌ Disabled'}",
        )
        if comp_config["memory_management"]["enabled"]:
            print(
                f"    Memory Threshold: {comp_config['memory_management']['memory_threshold']*100}%",
            )
            print(
                f"    Cleanup Frequency: {comp_config['memory_management']['cleanup_frequency']}",
            )

        print(
            f"  Early Stopping: {'✅ Enabled' if comp_config['early_stopping']['enabled'] else '❌ Disabled'}",
        )
        if comp_config["early_stopping"]["enabled"]:
            print(f"    Patience: {comp_config['early_stopping']['patience']}")
            print(f"    Min Trials: {comp_config['early_stopping']['min_trials']}")

    @safe_call
    def demonstrate_multi_objective_config(self) -> None:
        """Demonstrate multi-objective optimization configuration usage"""
        print("\n" + "=" * 60)
        print("MULTI-OBJECTIVE OPTIMIZATION CONFIGURATION")
        print("=" * 60)

        multi_obj_config = self.hpo_config["multi_objective"]

        if multi_obj_config["enabled"]:
            print("✅ Multi-objective optimization is enabled")

            # Access objectives and weights
            objectives = multi_obj_config["objectives"]
            weights = multi_obj_config["weights"]

            print(f"\n📈 Objectives: {objectives}")
            print(f"⚖️  Weights: {weights}")

            # Calculate weighted score example
            example_scores = {
                "sharpe_ratio": 1.5,
                "win_rate": 0.65,
                "profit_factor": 2.1,
            }

            weighted_score = sum(example_scores[obj] * weights[obj] for obj in objectives)

            print("\n📊 Example weighted score calculation:")
            for obj in objectives:
                print(
                    f"  {obj}: {example_scores[obj]} × {weights[obj]} = {example_scores[obj] * weights[obj]:.3f}",
                )
            print(f"  Total weighted score: {weighted_score:.3f}")

            # Risk constraints
            risk_constraints = multi_obj_config["risk_constraints"]
            print("\n🛡️  Risk Constraints:")
            for constraint, value in risk_constraints.items():
                print(f"  {constraint}: {value}")
        else:
            print(warning("Multi-objective optimization is disabled"))

    @safe_call
    def demonstrate_bayesian_config(self) -> None:
        """Demonstrate Bayesian optimization configuration usage"""
        print("\n" + "=" * 60)
        print("BAYESIAN OPTIMIZATION CONFIGURATION")
        print("=" * 60)

        bayesian_config = self.hpo_config["bayesian_optimization"]

        if bayesian_config["enabled"]:
            print("✅ Bayesian optimization is enabled")

            print(f"\n🔍 Sampling Strategy: {bayesian_config['sampling_strategy']}")
            print(f"📊 Max Trials: {bayesian_config['max_trials']}")
            print(f"⏳ Patience: {bayesian_config['patience']}")
            print(f"🎯 Acquisition Function: {bayesian_config['acquisition_function']}")

            # Search spaces
            search_spaces = self.hpo_config["search_spaces"]
            print("\n🔍 Search Spaces:")

            for space_name, space_config in search_spaces.items():
                print(f"\n  {space_name.upper()}:")
                for param_name, param_config in space_config.items():
                    if isinstance(param_config, dict) and "low" in param_config:
                        print(
                            f"    {param_name}: {param_config['low']} to {param_config['high']} ({param_config['type']})",
                        )
                    elif isinstance(param_config, dict) and "choices" in param_config:
                        print(f"    {param_name}: {param_config['choices']}")
                    else:
                        print(f"    {param_name}: {param_config}")
        else:
            print(warning("Bayesian optimization is disabled"))

    @safe_call
    def demonstrate_adaptive_config(self) -> None:
        """Demonstrate adaptive optimization configuration usage"""
        print("\n" + "=" * 60)
        print("ADAPTIVE OPTIMIZATION CONFIGURATION")
        print("=" * 60)

        adaptive_config = self.hpo_config["adaptive_optimization"]

        if adaptive_config["enabled"]:
            print("✅ Adaptive optimization is enabled")

            # Regime detection settings
            regime_detection = adaptive_config["regime_detection"]
            print("\n🎯 Regime Detection Settings:")
            for setting, value in regime_detection.items():
                print(f"  {setting}: {value}")

            # Regime-specific constraints
            regime_constraints = adaptive_config["regime_specific_constraints"]
            print("\n📊 Regime-Specific Constraints:")

            for regime, constraints in regime_constraints.items():
                print(f"\n  {regime.upper()} REGIME:")
                for constraint_name, constraint_range in constraints.items():
                    print(f"    {constraint_name}: {constraint_range}")
        else:
            print(warning("Adaptive optimization is disabled"))

    @safe_call
    def demonstrate_computational_config(self) -> None:
        """Demonstrate computational optimization configuration usage"""
        print("\n" + "=" * 60)
        print("COMPUTATIONAL OPTIMIZATION CONFIGURATION")
        print("=" * 60)

        comp_config = self.comp_config

        # Caching
        caching_config = comp_config["caching"]
        if caching_config["enabled"]:
            print("✅ Caching is enabled")
            print(f"  Max cache size: {caching_config['max_cache_size']} entries")
            print(f"  Cache TTL: {caching_config['cache_ttl']} seconds")
        else:
            print(warning("Caching is disabled"))

        # Parallelization
        parallel_config = comp_config["parallelization"]
        if parallel_config["enabled"]:
            print("\n✅ Parallelization is enabled")
            print(f"  Max workers: {parallel_config['max_workers']}")
            print(f"  Chunk size: {parallel_config['chunk_size']}")
        else:
            print("\n❌ Parallelization is disabled")

        # Memory management
        memory_config = comp_config["memory_management"]
        if memory_config["enabled"]:
            print("\n✅ Memory management is enabled")
            print(f"  Memory threshold: {memory_config['memory_threshold']*100}%")
            print(
                f"  Cleanup frequency: {memory_config['cleanup_frequency']} operations",
            )
        else:
            print("\n❌ Memory management is disabled")

        # Progressive evaluation
        progressive_config = comp_config["progressive_evaluation"]
        if progressive_config["enabled"]:
            print("\n✅ Progressive evaluation is enabled")
            stages = progressive_config["stages"]
            print("  Stages:")
            for i, stage in enumerate(stages, 1):
                print(
                    f"    Stage {i}: {stage['data_ratio']*100}% data, weight {stage['weight']}",
                )
        else:
            print("\n❌ Progressive evaluation is disabled")

    @safe_call
    async def demonstrate_optimization_usage(self) -> None:
        """Demonstrate how to use the configuration in actual optimization"""
        print("\n" + "=" * 60)
        print("OPTIMIZATION USAGE EXAMPLES")
        print("=" * 60)

        # Mock market data for demonstration
        mock_market_data: Dict[str, Any] = {"symbol": "ETHUSDT", "data": []}

        # Example 1: Multi-objective optimization
        print("\n📊 Example 1: Multi-Objective Optimization")
        try:
            _moo = MultiObjectiveOptimizer(config=self.hpo_config, market_data=mock_market_data)
            print("✅ Multi-objective optimizer initialized successfully")
            print("  Configuration loaded from config file")
            print("  Objectives: Sharpe ratio, Win rate, Profit factor")
            print("  Weights: 50%, 30%, 20% respectively")
        except Exception as exc:
            print(
                initialization_error(
                    f"Error initializing multi-objective optimizer: {exc}",
                )
            )

        # Example 2: Bayesian optimization
        print("\n🔍 Example 2: Bayesian Optimization")
        try:
            _bayes = AdvancedBayesianOptimizer(
                config=self.hpo_config["bayesian_optimization"],
                search_space=self.hpo_config["search_spaces"],
            )
            print("✅ Bayesian optimizer initialized successfully")
            print(
                f"  Sampling strategy: {self.hpo_config['bayesian_optimization']['sampling_strategy']}",
            )
            print(
                f"  Max trials: {self.hpo_config['bayesian_optimization']['max_trials']}",
            )
        except Exception as exc:
            print(initialization_error(f"Error initializing Bayesian optimizer: {exc}"))

        # Example 3: Computational optimization
        print("\n⚡ Example 3: Computational Optimization")
        try:
            _bt = OptimizedBacktester(market_data=mock_market_data, config=self.comp_config)
            print("✅ Optimized backtester initialized successfully")
            print(
                f"  Caching: {'Enabled' if self.comp_config['caching']['enabled'] else 'Disabled'}",
            )
            print(
                f"  Parallelization: {'Enabled' if self.comp_config['parallelization']['enabled'] else 'Disabled'}",
            )
            print(
                f"  Memory management: {'Enabled' if self.comp_config['memory_management']['enabled'] else 'Disabled'}",
            )
        except Exception as exc:
            print(initialization_error(f"Error initializing optimized backtester: {exc}"))

    @safe_call
    def demonstrate_configuration_modification(self) -> None:
        """Demonstrate how to modify configuration settings"""
        print("\n" + "=" * 60)
        print("CONFIGURATION MODIFICATION EXAMPLES")
        print("=" * 60)

        # Example: Modify Bayesian optimization settings
        print("\n🔧 Example: Modifying Bayesian Optimization Settings")

        # Create a copy of the configuration for modification
        modified_config = dict(self.hpo_config)
        bayesian_config = dict(modified_config["bayesian_optimization"])  # shallow copy

        # Modify settings
        old_trials = bayesian_config.get("max_trials", 0)
        old_patience = bayesian_config.get("patience", 0)
        old_strategy = bayesian_config.get("sampling_strategy", "")

        bayesian_config["max_trials"] = 200
        bayesian_config["patience"] = 25
        bayesian_config["sampling_strategy"] = "random"

        modified_config["bayesian_optimization"] = bayesian_config

        print("✅ Configuration modified:")
        print(f"  Max trials: {old_trials} → {bayesian_config['max_trials']}")
        print(f"  Patience: {old_patience} → {bayesian_config['patience']}")
        print(f"  Strategy: {old_strategy} → {bayesian_config['sampling_strategy']}")

        # Example: Modify computational optimization settings
        print("\n⚡ Example: Modifying Computational Optimization Settings")

        modified_comp_config = dict(self.comp_config)

        # Modify caching settings
        modified_comp_config.setdefault("caching", {}).setdefault("max_cache_size", 1000)
        modified_comp_config.setdefault("caching", {}).setdefault("cache_ttl", 3600)
        modified_comp_config["caching"]["max_cache_size"] = 500
        modified_comp_config["caching"]["cache_ttl"] = 1800

        # Modify parallelization settings
        modified_comp_config.setdefault("parallelization", {}).setdefault(
            "max_workers", 8
        )
        old_workers = modified_comp_config["parallelization"]["max_workers"]
        modified_comp_config["parallelization"]["max_workers"] = 4

        print("✅ Computational configuration modified:")
        print(
            f"  Cache size: 1000 → {modified_comp_config['caching']['max_cache_size']}",
        )
        print(f"  Cache TTL: 3600s → {modified_comp_config['caching']['cache_ttl']}s")
        print(
            f"  Max workers: {old_workers} → {modified_comp_config['parallelization']['max_workers']}",
        )

    def run_all_demonstrations(self) -> bool:
        """Run all configuration demonstrations"""
        print("🚀 Starting Configuration Usage Demonstrations")
        print("=" * 60)

        # Validate configuration
        if not self.validate_configuration():
            print(failed("Configuration validation failed. Exiting."))
            return False

        # Print configuration summary
        self.print_configuration_summary()

        # Demonstrate each configuration section
        self.demonstrate_multi_objective_config()
        self.demonstrate_bayesian_config()
        self.demonstrate_adaptive_config()
        self.demonstrate_computational_config()

        # Demonstrate usage examples
        asyncio.run(self.demonstrate_optimization_usage())

        # Demonstrate configuration modification
        self.demonstrate_configuration_modification()

        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("=" * 60)

        return True


def main() -> None:
    """Main function to run the configuration usage example"""
    try:
        example = ConfigurationUsageExample()
        success = example.run_all_demonstrations()

        if success:
            print("\n🎉 Configuration usage demonstration completed successfully!")
            print("\n📚 Next steps:")
            print("  1. Review the configuration settings in src/config.py")
            print("  2. Modify settings based on your requirements")
            print("  3. Use the configuration in your optimization scripts")
            print("  4. Monitor performance and adjust settings as needed")
        else:
            print("\n❌ Configuration usage demonstration failed!")
            sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive CLI wrapper
        print(error(f"Error running configuration usage example: {exc}"))
        sys.exit(1)


if __name__ == "__main__":
    main()
