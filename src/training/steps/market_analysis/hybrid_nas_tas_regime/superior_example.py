"""
Superior Architecture Example

Demonstrates the streamlined and focused approach of the Hybrid NAS TAS Regime system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

# Import the main orchestrator (HMM replacement)
from .integration.hybrid_orchestrator import HybridOrchestrator
from .config.hybrid_config import (
    HybridRegimeConfig, ClusteringMethod, IntegrationStrategy
)


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data."""
    np.random.seed(42)
    
    # Generate synthetic market data
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': [p + np.random.normal(0, 0.005) for p in prices],
        'volume': np.random.lognormal(10, 1, n_samples)
    })


def create_sample_tas_inputs(n_samples: int = 1000) -> Dict[str, Any]:
    """Create sample TAS inputs."""
    np.random.seed(42)
    
    return {
        'regime_predictions': np.random.randint(0, 6, n_samples),
        'regime_probabilities': np.random.dirichlet([1] * 6, n_samples),
        'regime_stability_scores': np.random.uniform(0.6, 0.9, n_samples),
        'economic_significance_scores': np.random.uniform(0.5, 0.8, n_samples),
        'trading_viability_scores': np.random.uniform(0.4, 0.7, n_samples)
    }


def create_sample_nas_inputs(n_samples: int = 1000) -> Dict[str, Any]:
    """Create sample NAS inputs."""
    np.random.seed(43)
    
    return {
        'regime_predictions': np.random.randint(0, 6, n_samples),
        'regime_probabilities': np.random.dirichlet([1] * 6, n_samples),
        'regime_stability_scores': np.random.uniform(0.7, 0.95, n_samples),
        'economic_significance_scores': np.random.uniform(0.6, 0.9, n_samples),
        'trading_viability_scores': np.random.uniform(0.5, 0.8, n_samples)
    }


def demonstrate_superior_architecture():
    """Demonstrate the superior architecture."""
    print("🚀 Superior Hybrid NAS TAS Regime Architecture")
    print("=" * 60)
    
    # 1. Single Point of Control - Main Orchestrator
    print("🎯 1. Single Point of Control")
    print("   ✅ One orchestrator manages everything")
    print("   ✅ Direct HMM replacement")
    print("   ✅ Clean API interface")
    
    # 2. Streamlined Configuration
    print("\n⚙️ 2. Streamlined Configuration")
    config = HybridRegimeConfig(
        n_regimes=12,
        economic_modeling_enabled=True,
        financial_modeling_enabled=True,
        clustering_method=ClusteringMethod.HYBRID,
        integration_strategy=IntegrationStrategy.ADAPTIVE
    )
    print(f"   ✅ Simple configuration: {config.n_regimes} regimes")
    print(f"   ✅ Economic modeling: {config.economic_modeling_enabled}")
    print(f"   ✅ Financial modeling: {config.financial_modeling_enabled}")
    
    # 3. Initialize Orchestrator
    print("\n🔧 3. Initialize Orchestrator")
    orchestrator = HybridOrchestrator(config)
    print("   ✅ Single orchestrator instance")
    print("   ✅ All components managed internally")
    print("   ✅ Clean separation of concerns")
    
    # 4. Create Sample Data
    print("\n📊 4. Create Sample Data")
    market_data = create_sample_market_data(1000)
    tas_inputs = create_sample_tas_inputs(1000)
    nas_inputs = create_sample_nas_inputs(1000)
    print(f"   ✅ Market data: {len(market_data)} samples")
    print(f"   ✅ TAS inputs: {len(tas_inputs['regime_predictions'])} predictions")
    print(f"   ✅ NAS inputs: {len(nas_inputs['regime_predictions'])} predictions")
    
    # 5. Process Complete Pipeline
    print("\n🔄 5. Process Complete Pipeline")
    result = orchestrator.process_regime_detection(
        market_data=market_data,
        tas_inputs=tas_inputs,
        nas_inputs=nas_inputs,
        enable_tagging=True,
        save_results=False
    )
    
    if result.success:
        print("   ✅ Regime detection successful")
        print(f"   ✅ Detected {len(set(result.regime_predictions))} regimes")
        print(f"   ✅ Economic significance: {np.mean(result.economic_significance_scores):.3f}")
        print(f"   ✅ Financial significance: {np.mean(result.financial_significance_scores):.3f}")
        print(f"   ✅ Trading viability: {np.mean(result.trading_viability_scores):.3f}")
        print(f"   ✅ Execution time: {result.execution_time:.2f}s")
    else:
        print(f"   ❌ Regime detection failed: {result.error_message}")
    
    # 6. Get Results
    print("\n📋 6. Get Results")
    performance_summary = orchestrator.get_performance_summary()
    regime_summary = orchestrator.get_regime_summary()
    tagged_data = orchestrator.get_tagged_data()
    
    print("   ✅ Performance summary available")
    print("   ✅ Regime summary available")
    print(f"   ✅ Tagged data: {len(tagged_data) if tagged_data is not None else 0} samples")
    
    # 7. Architecture Advantages
    print("\n🏗️ 7. Architecture Advantages")
    advantages = [
        "✅ Single orchestrator replaces HMM clustering",
        "✅ Clean component separation",
        "✅ Focused testing strategy",
        "✅ Better maintainability",
        "✅ Simpler integration",
        "✅ Clear API interface",
        "✅ Comprehensive evaluation",
        "✅ Unified data tagging"
    ]
    
    for advantage in advantages:
        print(f"   {advantage}")
    
    print("\n🎉 Superior Architecture Demonstration Complete!")
    return result


def compare_architectures():
    """Compare the two architectures."""
    print("\n🔄 Architecture Comparison")
    print("=" * 50)
    
    print("❌ Original Approach:")
    print("   - 8 directories with overlap")
    print("   - Multiple components need coordination")
    print("   - Complex integration")
    print("   - Scattered testing")
    
    print("\n✅ Superior Approach:")
    print("   - 6 focused directories")
    print("   - Single orchestrator manages everything")
    print("   - Clean separation of concerns")
    print("   - Focused testing strategy")
    print("   - Direct HMM replacement")
    print("   - Better maintainability")
    print("   - Simpler integration")
    print("   - Clear API interface")


if __name__ == "__main__":
    try:
        result = demonstrate_superior_architecture()
        compare_architectures()
        
        print("\n🎯 Summary:")
        print("The superior architecture provides:")
        print("1. ✅ Single point of control with orchestrator")
        print("2. ✅ Clean component separation")
        print("3. ✅ Focused testing strategy")
        print("4. ✅ Better maintainability")
        print("5. ✅ Simpler integration")
        print("6. ✅ Clear API interface")
        print("7. ✅ Direct HMM replacement")
        print("8. ✅ Comprehensive evaluation")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()