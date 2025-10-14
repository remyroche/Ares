"""
Ares Launcher Integration Example for Unified Data-Driven Pipeline

This example shows how to integrate the unified pipeline with the existing
ares_launcher.py commands:

- --unified-pipeline-analyst
- --unified-pipeline-tactician  
- --unified-pipeline-analyst-short
- --unified-pipeline-tactician-long

The difference between tactician and analyst is in the labels used to qualify the financial data.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.pre_training.unified_data_driven_pipeline.unified_pipeline_commands import (
    create_unified_pipeline_command_handler,
    handle_unified_pipeline_analyst,
    handle_unified_pipeline_tactician,
    handle_unified_pipeline_analyst_short,
    handle_unified_pipeline_tactician_long
)


def example_command_handlers():
    """Example of using command handlers directly."""
    print("=" * 80)
    print("UNIFIED PIPELINE COMMAND HANDLERS")
    print("=" * 80)
    
    # Create command handler
    handler = create_unified_pipeline_command_handler()
    
    # Example 1: Analyst mode
    print("\n1. Analyst Mode (--unified-pipeline-analyst):")
    print("   Purpose: 'Should we trade?' based on expected PnL > fees + slippage")
    
    analyst_pipeline = handler.handle_analyst_command(
        symbol="ETHUSDT",
        intensity="blank"  # 25% intensity
    )
    
    print(f"   Pipeline created: {type(analyst_pipeline).__name__}")
    print(f"   Labeling type: analyst")
    print(f"   Direction: long")
    
    # Example 2: Tactician mode
    print("\n2. Tactician Mode (--unified-pipeline-tactician):")
    print("   Purpose: Direction/magnitude based on max favorable/adverse excursion")
    
    tactician_pipeline = handler.handle_tactician_command(
        symbol="ETHUSDT",
        intensity="blank"  # 25% intensity
    )
    
    print(f"   Pipeline created: {type(tactician_pipeline).__name__}")
    print(f"   Labeling type: tactician")
    print(f"   Direction: long")
    
    # Example 3: Analyst Short mode
    print("\n3. Analyst Short Mode (--unified-pipeline-analyst-short):")
    print("   Purpose: Short position analysis using analyst labeling")
    
    analyst_short_pipeline = handler.handle_analyst_short_command(
        symbol="ETHUSDT",
        intensity="light"  # 10% intensity for faster processing
    )
    
    print(f"   Pipeline created: {type(analyst_short_pipeline).__name__}")
    print(f"   Labeling type: analyst")
    print(f"   Direction: short")
    
    # Example 4: Tactician Long mode
    print("\n4. Tactician Long Mode (--unified-pipeline-tactician-long):")
    print("   Purpose: Long tactical analysis using tactician labeling")
    
    tactician_long_pipeline = handler.handle_tactician_long_command(
        symbol="ETHUSDT",
        intensity="full"  # 100% intensity for maximum performance
    )
    
    print(f"   Pipeline created: {type(tactician_long_pipeline).__name__}")
    print(f"   Labeling type: tactician")
    print(f"   Direction: long")
    
    # Cleanup
    analyst_pipeline.cleanup()
    tactician_pipeline.cleanup()
    analyst_short_pipeline.cleanup()
    tactician_long_pipeline.cleanup()


def example_convenience_functions():
    """Example of using convenience functions."""
    print("\n" + "=" * 80)
    print("CONVENIENCE FUNCTIONS")
    print("=" * 80)
    
    # Example 1: Direct function calls
    print("\n1. Using convenience functions:")
    
    # Analyst pipeline
    analyst_pipeline = handle_unified_pipeline_analyst(
        symbol="BTCUSDT",
        intensity="blank"
    )
    print(f"   Analyst pipeline for BTCUSDT: {type(analyst_pipeline).__name__}")
    
    # Tactician pipeline
    tactician_pipeline = handle_unified_pipeline_tactician(
        symbol="BTCUSDT",
        intensity="light"
    )
    print(f"   Tactician pipeline for BTCUSDT: {type(tactician_pipeline).__name__}")
    
    # Cleanup
    analyst_pipeline.cleanup()
    tactician_pipeline.cleanup()


def example_intensity_levels():
    """Example of different intensity levels with each command."""
    print("\n" + "=" * 80)
    print("INTENSITY LEVELS WITH COMMANDS")
    print("=" * 80)
    
    handler = create_unified_pipeline_command_handler()
    
    intensities = ["light", "blank", "full"]
    commands = ["analyst", "tactician"]
    
    print("\nIntensity comparison for different commands:")
    print("-" * 60)
    print(f"{'Command':<12} {'Intensity':<10} {'Description'}")
    print("-" * 60)
    
    for command in commands:
        for intensity in intensities:
            # Get command info
            command_info = handler.get_command_info(command)
            description = command_info.get('description', 'Unknown')
            
            print(f"{command:<12} {intensity:<10} {description[:40]}...")
    
    print("-" * 60)


def example_command_info():
    """Example of getting command information."""
    print("\n" + "=" * 80)
    print("COMMAND INFORMATION")
    print("=" * 80)
    
    handler = create_unified_pipeline_command_handler()
    
    # List all available commands
    commands = handler.list_available_commands()
    
    print("\nAvailable unified pipeline commands:")
    print("-" * 50)
    
    for command_type, info in commands.items():
        print(f"\n{command_type.upper()}:")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Labeling Type: {info.get('labeling_type', 'N/A')}")
        print(f"  Direction: {info.get('direction', 'N/A')}")
        print(f"  Use Case: {info.get('use_case', 'N/A')}")


def example_ares_launcher_integration():
    """Example of how to integrate with ares_launcher.py."""
    print("\n" + "=" * 80)
    print("ARES LAUNCHER INTEGRATION")
    print("=" * 80)
    
    print("\nTo integrate with existing ares_launcher.py, add these functions:")
    print("-" * 60)
    
    integration_code = '''
# Add to ares_launcher.py

from src.training.steps.pre_training.unified_data_driven_pipeline.unified_pipeline_commands import (
    handle_unified_pipeline_analyst,
    handle_unified_pipeline_tactician,
    handle_unified_pipeline_analyst_short,
    handle_unified_pipeline_tactician_long
)

def unified_pipeline_analyst(args):
    """Handle --unified-pipeline-analyst command."""
    symbol = getattr(args, 'symbol', 'ETHUSDT')
    intensity = getattr(args, 'intensity', 'blank')
    
    pipeline = handle_unified_pipeline_analyst(
        symbol=symbol,
        intensity=intensity
    )
    
    # Process data with pipeline
    # result = pipeline.process(data, targets, timeframe="15m")
    
    return pipeline

def unified_pipeline_tactician(args):
    """Handle --unified-pipeline-tactician command."""
    symbol = getattr(args, 'symbol', 'ETHUSDT')
    intensity = getattr(args, 'intensity', 'blank')
    
    pipeline = handle_unified_pipeline_tactician(
        symbol=symbol,
        intensity=intensity
    )
    
    # Process data with pipeline
    # result = pipeline.process(data, targets, timeframe="15m")
    
    return pipeline

def unified_pipeline_analyst_short(args):
    """Handle --unified-pipeline-analyst-short command."""
    symbol = getattr(args, 'symbol', 'ETHUSDT')
    intensity = getattr(args, 'intensity', 'blank')
    
    pipeline = handle_unified_pipeline_analyst_short(
        symbol=symbol,
        intensity=intensity
    )
    
    # Process data with pipeline
    # result = pipeline.process(data, targets, timeframe="15m")
    
    return pipeline

def unified_pipeline_tactician_long(args):
    """Handle --unified-pipeline-tactician-long command."""
    symbol = getattr(args, 'symbol', 'ETHUSDT')
    intensity = getattr(args, 'intensity', 'blank')
    
    pipeline = handle_unified_pipeline_tactician_long(
        symbol=symbol,
        intensity=intensity
    )
    
    # Process data with pipeline
    # result = pipeline.process(data, targets, timeframe="15m")
    
    return pipeline
    '''
    
    print(integration_code)
    
    print("\nCommand line usage examples:")
    print("-" * 40)
    print("# Analyst mode (long positions, 15m timeframe)")
    print("python ares_launcher.py --unified-pipeline-analyst --symbol ETHUSDT")
    print("")
    print("# Tactician mode (long positions, 15m timeframe)")
    print("python ares_launcher.py --unified-pipeline-tactician --symbol ETHUSDT")
    print("")
    print("# Specific directions still available")
    print("python ares_launcher.py --unified-pipeline-analyst-short --symbol ETHUSDT")
    print("python ares_launcher.py --unified-pipeline-tactician-long --symbol ETHUSDT")
    print("")
    print("# With intensity settings (if added to ares_launcher.py)")
    print("python ares_launcher.py --unified-pipeline-analyst --symbol ETHUSDT --intensity light")
    print("python ares_launcher.py --unified-pipeline-tactician --symbol ETHUSDT --intensity full")


def example_labeling_differences():
    """Example showing the differences between tactician and analyst labeling."""
    print("\n" + "=" * 80)
    print("TACTICIAN VS ANALYST LABELING DIFFERENCES")
    print("=" * 80)
    
    print("\nTACTICIAN LABELING:")
    print("-" * 30)
    print("Purpose: Direction/magnitude based on max favorable/adverse excursion")
    print("Focus: Short-term tactical decisions")
    print("Labels: Entry/exit signals with magnitude")
    print("Use Case: When to enter/exit positions")
    print("Time Horizon: Short-term (minutes to hours)")
    
    print("\nANALYST LABELING:")
    print("-" * 30)
    print("Purpose: 'Should we trade?' based on expected PnL > fees + slippage")
    print("Focus: Long-term position analysis")
    print("Labels: Binary decision (trade/not trade)")
    print("Use Case: Whether to take a position")
    print("Time Horizon: Medium-term (hours to days)")
    
    print("\nKEY DIFFERENCES:")
    print("-" * 30)
    print("1. Tactician: 'When and how much to trade'")
    print("2. Analyst: 'Whether to trade at all'")
    print("3. Tactician: Direction and magnitude")
    print("4. Analyst: Binary decision")
    print("5. Tactician: Short-term focus")
    print("6. Analyst: Long-term focus")


def main():
    """Run all ares_launcher integration examples."""
    print("ARES LAUNCHER INTEGRATION EXAMPLES")
    print("=" * 80)
    
    try:
        example_command_handlers()
        example_convenience_functions()
        example_intensity_levels()
        example_command_info()
        example_ares_launcher_integration()
        example_labeling_differences()
        
        print("\n" + "=" * 80)
        print("ALL ARES LAUNCHER INTEGRATION EXAMPLES COMPLETED!")
        print("=" * 80)
        
        print("\nNext steps:")
        print("1. Add the integration code to ares_launcher.py")
        print("2. Test with existing commands:")
        print("   python ares_launcher.py --unified-pipeline-analyst --symbol ETHUSDT")
        print("   python ares_launcher.py --unified-pipeline-tactician --symbol ETHUSDT")
        print("3. Optionally add --intensity parameter to ares_launcher.py")
        
    except Exception as e:
        print(f"\nERROR: Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()