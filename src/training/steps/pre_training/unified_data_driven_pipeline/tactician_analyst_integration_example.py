#!/usr/bin/env python3
"""
Example usage of Tactician/Analyst labeling integration in UnifiedDataDrivenPipeline.

This example shows how to configure and use the tactician/analyst labeling system
instead of the traditional triple barrier labeling.
"""

from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import (
    UnifiedPipelineConfig, 
    create_default_config
)

def create_analyst_config():
    """Create configuration for Analyst labeling."""
    config = create_default_config()
    
    # Configure for analyst labeling
    config.labeling_system = "tactician_analyst"
    config.labeling_type = "analyst"
    config.enable_labeling_optimization = True
    config.labeling_quality_threshold = 0.7
    
    return config

def create_tactician_config():
    """Create configuration for Tactician labeling."""
    config = create_default_config()
    
    # Configure for tactician labeling
    config.labeling_system = "tactician_analyst"
    config.labeling_type = "tactician"
    config.enable_labeling_optimization = True
    config.labeling_quality_threshold = 0.7
    
    return config

def create_triple_barrier_config():
    """Create configuration for triple barrier labeling (fallback)."""
    config = create_default_config()
    
    # Configure for triple barrier labeling
    config.labeling_system = "triple_barrier"
    config.enable_labeling_optimization = True
    config.labeling_quality_threshold = 0.7
    
    return config

def main():
    """Demonstrate different labeling configurations."""
    print("🏷️ Tactician/Analyst Labeling Integration Example")
    print("=" * 60)
    
    # Example 1: Analyst labeling
    print("\n1. Analyst Labeling Configuration:")
    analyst_config = create_analyst_config()
    print(f"   - Labeling System: {analyst_config.labeling_system}")
    print(f"   - Labeling Type: {analyst_config.labeling_type}")
    print(f"   - Quality Threshold: {analyst_config.labeling_quality_threshold}")
    print("   - Purpose: 'Should we trade?' based on expected PnL > fees + slippage")
    
    # Example 2: Tactician labeling
    print("\n2. Tactician Labeling Configuration:")
    tactician_config = create_tactician_config()
    print(f"   - Labeling System: {tactician_config.labeling_system}")
    print(f"   - Labeling Type: {tactician_config.labeling_type}")
    print(f"   - Quality Threshold: {tactician_config.labeling_quality_threshold}")
    print("   - Purpose: Direction/magnitude based on max favorable/adverse excursion")
    
    # Example 3: Triple barrier fallback
    print("\n3. Triple Barrier Fallback Configuration:")
    triple_barrier_config = create_triple_barrier_config()
    print(f"   - Labeling System: {triple_barrier_config.labeling_system}")
    print(f"   - Quality Threshold: {triple_barrier_config.labeling_quality_threshold}")
    print("   - Purpose: Traditional triple barrier method (fallback)")
    
    print("\n" + "=" * 60)
    print("✅ Configuration examples created successfully!")
    print("\nTo use these configurations:")
    print("   from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline")
    print("   pipeline = UnifiedDataDrivenPipeline(analyst_config)  # or tactician_config, triple_barrier_config")

if __name__ == "__main__":
    main()