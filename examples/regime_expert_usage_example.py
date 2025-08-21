# examples/regime_expert_usage_example.py

"""
Example demonstrating the new regime expert orchestrator system.
This shows how to use composite_cluster_id based regime detection with specialized experts.
"""

import asyncio
import yaml

from src.analyst.regime_expert_orchestrator import (
    RegimeExpertOrchestrator,
    get_regime_expert_decision,
)

async def example_basic_regime_detection():
    """Example of basic regime detection using composite_cluster_id."""

    # Load configuration
    with open("src/config/regime_mapping_config.yaml") as f:
        config , yaml.safe_load(f)

    # Initialize orchestrator
    orchestrator = RegimeExpertOrchestrator(config)
    await orchestrator.initialize()

    # Get current regime information
    regime_info = await orchestrator.get_current_regime_info(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
    )

    if regime_info:
        print("Current Regime Info:")
        print(f"  Cluster ID: {regime_info['cluster_id']}")
        print(f"  Regime Name: {regime_info['regime_name']}")
        print(f"  Confidence: {regime_info['confidence']:.3f}")
        print(
            f"  Expert Type: {type(regime_info['expert']).__name__ if regime_info['expert'] else 'None'}",
        )
    else:
        print("Could not determine current regime")

async def example_regime_expert_prediction():
    """Example of getting predictions from regime experts."""

    with open("src/config/regime_mapping_config.yaml") as f:
        config = yaml.safe_load(f)

    orchestrator = RegimeExpertOrchestrator(config)
    await orchestrator.initialize()

    # Get regime info
    regime_info = await orchestrator.get_current_regime_info(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
    )

    if regime_info and regime_info["expert"]:
        # Get prediction from the regime expert
        prediction = await orchestrator.get_regime_expert_prediction(
            current_features=None,  # Would be actual features in real usage
            regime_info=regime_info,
        )

        if prediction:
            print("Regime Expert Prediction:")
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Regime: {prediction['regime']}")
            print(f"  Cluster ID: {prediction['cluster_id']}")

async def example_two_tier_decision_system():
    """Example of the two-tier decision system with Step 9.5 and Step 10 integration."""

    with open("src/config/regime_mapping_config.yaml") as f:
        config = yaml.safe_load(f)

    orchestrator = RegimeExpertOrchestrator(config)
    await orchestrator.initialize()

    # Mock Step 9.5 prediction (in real usage, this would come from Step 9.5)
    step9_5_prediction = {
        "regime_transition_prob": 0.3,
        "price_direction": "UP",
        "tpsl_probabilities": {"profit_target": 0.6, "stop_loss": 0.2},
        "confidence": 0.7,
        "current_features": None,  # Would be actual features
    }

    # Mock Step 10 prediction (in real usage, this would come from Step 10)
    step10_prediction = {
        "path_class": "beginning_of_trend",
        "optimal_timing": 5,
        "confidence": 0.8,
        "current_features": None,  # Would be actual features
    }

    # Get two-tier decision
    decision = await orchestrator.get_two_tier_decision(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
        step9_5_prediction=step9_5_prediction,
        step10_prediction=step10_prediction,
    )

    if decision:
        print("Two-Tier Decision Result:")
        print(f"  Regime: {decision['regime_info']['regime_name']}")
        print(f"  Cluster ID: {decision['regime_info']['cluster_id']}")

        strategic = decision["strategic_decision"]
        print(
            f"  Strategic Decision: {strategic['prediction']} (confidence: {strategic['confidence']:.3f})",
        )

        final = decision["final_decision"]
        print(f"  Final Decision: {final['action']}")
        print(f"  Timing: {final['timing']}")
        print(f"  Reason: {final['reason']}")
        print(f"  Confidence: {final['confidence']:.3f}")

async def example_continuous_monitoring():
    """Example of continuous monitoring for regime changes."""

    with open("src/config/regime_mapping_config.yaml") as f:
        config = yaml.safe_load(f)

    orchestrator = RegimeExpertOrchestrator(config)
    await orchestrator.initialize()

    print("Starting continuous monitoring (will run for 5 minutes)...")

    # Run monitoring for a limited time in this example
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 300:  # 5 minutes
        decision = await orchestrator.get_two_tier_decision(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
        )

        if decision and decision["final_decision"]["action"] != "HOLD":
            print(f"Trading Signal Detected: {decision['final_decision']}")

        await asyncio.sleep(60)  # Check every minute

    print("Continuous monitoring completed")

async def example_cluster_mapping():
    """Example showing how cluster IDs map to regime names."""

    with open("src/config/regime_mapping_config.yaml") as f:
        config = yaml.safe_load(f)

    orchestrator = RegimeExpertOrchestrator(config)

    print("Cluster ID to Regime Mapping:")
    for cluster_id in range(6):
        regime_name = orchestrator.get_current_regime_from_cluster(cluster_id)
        expert = orchestrator.get_regime_expert(cluster_id)
        print(
            f"  Cluster {cluster_id} -> {regime_name} (Expert: {type(expert).__name__ if expert else 'None'})",
        )

async def example_convenience_function():
    """Example using the convenience function for quick regime decisions."""

    with open("src/config/regime_mapping_config.yaml") as f:
        config = yaml.safe_load(f)

    # Use the convenience function
    decision = await get_regime_expert_decision(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
        config, config = )

    if decision:
        print("Quick Decision Result:")
        print(f"  Regime: {decision['regime_info']['regime_name']}")
        print(f"  Final Action: {decision['final_decision']['action']}")
        print(f"  Confidence: {decision['final_decision']['confidence']:.3f}")

async def main():
    """Run all examples."""
    print("=== Regime Expert Orchestrator Examples ===\n")

    print("1. Basic Regime Detection:")
    await example_basic_regime_detection()
    print()

    print("2. Regime Expert Prediction:")
    await example_regime_expert_prediction()
    print()

    print("3. Cluster ID Mapping:")
    await example_cluster_mapping()
    print()

    print("4. Two-Tier Decision System:")
    await example_two_tier_decision_system()
    print()

    print("5. Convenience Function:")
    await example_convenience_function()
    print()

    print("6. Continuous Monitoring (5 minutes):")
    await example_continuous_monitoring()
    print()

    print("All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
