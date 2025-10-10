#!/usr/bin/env python3
"""
Enhanced Position Test Runner

CLI tool with configuration file support for position testing.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_position_test_suite import (
    EnhancedPositionTester, PositionTestConfig, 
    PositionSide, OrderType
)

class PositionTestRunner:
    """Enhanced position test runner with configuration support."""
    
    def __init__(self):
        self.config_file = Path("position_test_configs.json")
        self.configs = {}
        self.load_configs()
    
    def load_configs(self) -> None:
        """Load test configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.configs = data.get('position_test_configs', {})
                    self.default_config = data.get('default_config', 'multi_symbol')
            except Exception as e:
                print(f"⚠️ Warning: Could not load config file: {e}")
                self.configs = {}
                self.default_config = 'multi_symbol'
        else:
            print("ℹ️ No config file found, using defaults")
            self.configs = {}
            self.default_config = 'multi_symbol'
    
    def list_configs(self) -> None:
        """List available test configurations."""
        print("📋 Available Position Test Configurations:")
        print("=" * 60)
        
        if not self.configs:
            print("No configurations found in position_test_configs.json")
            return
        
        for name, config in self.configs.items():
            print(f"\n🔧 {name}:")
            print(f"   Description: {config.get('description', 'N/A')}")
            print(f"   Symbols: {', '.join(config.get('symbols', []))}")
            print(f"   Sizes: {', '.join(map(str, config.get('position_sizes', [])))}")
            print(f"   Sides: {', '.join(config.get('sides', []))}")
            print(f"   Max Positions: {config.get('max_positions', 'N/A')}")
            print(f"   Cleanup: {config.get('cleanup_positions', 'N/A')}")
            print(f"   Type: {'Perpetuals' if config.get('test_perpetuals', True) else 'Spot'}")
    
    def create_test_config(self, config_name: Optional[str] = None, 
                          overrides: Optional[Dict[str, Any]] = None) -> PositionTestConfig:
        """Create a test configuration from file or defaults."""
        
        # Start with default config
        if config_name and config_name in self.configs:
            config_data = self.configs[config_name].copy()
        elif self.default_config in self.configs:
            config_data = self.configs[self.default_config].copy()
        else:
            config_data = {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "position_sizes": [0.001, 0.01],
                "sides": ["long", "short"],
                "max_positions": 4,
                "cleanup_positions": True,
                "test_perpetuals": True
            }
        
        # Apply overrides
        if overrides:
            config_data.update(overrides)
        
        # Convert side strings to enums
        side_enums = []
        for side in config_data.get('sides', []):
            if side.lower() == 'long':
                side_enums.append(PositionSide.LONG)
            elif side.lower() == 'short':
                side_enums.append(PositionSide.SHORT)
        
        # Create PositionTestConfig object
        return PositionTestConfig(
            symbols=config_data.get('symbols', ['BTCUSDT']),
            position_sizes=config_data.get('position_sizes', [0.001]),
            sides=side_enums,
            test_perpetuals=config_data.get('test_perpetuals', True),
            test_spot=not config_data.get('test_perpetuals', True),
            max_positions=config_data.get('max_positions', 4),
            cleanup_positions=config_data.get('cleanup_positions', True)
        )
    
    async def run_tests(self, config: PositionTestConfig, output_file: Optional[str] = None) -> bool:
        """Run the position tests with the given configuration."""
        print(f"🚀 Running Position Tests")
        print(f"📊 Symbols: {', '.join(config.symbols)}")
        print(f"💰 Sizes: {', '.join(map(str, config.position_sizes))}")
        print(f"📈 Sides: {', '.join(s.value for s in config.sides)}")
        print(f"🔢 Max Positions: {config.max_positions}")
        print(f"🧹 Cleanup: {config.cleanup_positions}")
        print(f"📊 Type: {'Perpetuals' if config.test_perpetuals else 'Spot'}")
        print("=" * 60)
        
        tester = EnhancedPositionTester(config)
        
        try:
            results = await tester.run_all_tests()
            
            # Save results to file if requested
            if output_file:
                self.save_results(results, output_file, config)
            
            # Return success status
            return sum(1 for r in results if not r.success) == 0
            
        except Exception as e:
            print(f"❌ Position tests failed: {e}")
            return False
        finally:
            await tester.cleanup()
    
    def save_results(self, results, output_file: str, config: PositionTestConfig) -> None:
        """Save test results to a JSON file."""
        try:
            output_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "config": {
                    "symbols": config.symbols,
                    "position_sizes": config.position_sizes,
                    "sides": [s.value for s in config.sides],
                    "max_positions": config.max_positions,
                    "cleanup_positions": config.cleanup_positions,
                    "test_perpetuals": config.test_perpetuals
                },
                "results": [
                    {
                        "operation": r.operation,
                        "success": r.success,
                        "duration": r.duration,
                        "error": r.error,
                        "data": r.data
                    } for r in results
                ],
                "summary": {
                    "total_tests": len(results),
                    "passed_tests": sum(1 for r in results if r.success),
                    "failed_tests": sum(1 for r in results if not r.success),
                    "success_rate": (sum(1 for r in results if r.success) / len(results) * 100) if results else 0
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"📄 Results saved to {output_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save results to {output_file}: {e}")

async def main():
    """Main entry point for the enhanced position test runner."""
    parser = argparse.ArgumentParser(
        description="Enhanced Position Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configurations
  python3 run_position_tests.py --list

  # Run with a specific configuration
  python3 run_position_tests.py --config basic_long

  # Run with overrides
  python3 run_position_tests.py --config multi_symbol --symbols BTCUSDT ETHUSDT --sizes 0.001 0.01

  # Run with custom settings
  python3 run_position_tests.py --symbols BTCUSDT --sides long short --sizes 0.001 0.01

  # Save results to file
  python3 run_position_tests.py --config comprehensive --output position_results.json
        """
    )
    
    # Configuration options
    parser.add_argument('--config', '-c',
                       help='Use a predefined configuration from position_test_configs.json')
    
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='List available configurations')
    
    # Test options
    parser.add_argument('--symbols', '-s',
                       nargs='+',
                       help='Trading symbols to test')
    
    parser.add_argument('--sizes', '-z',
                       nargs='+',
                       type=float,
                       help='Position sizes to test')
    
    parser.add_argument('--sides', '-d',
                       nargs='+',
                       choices=['long', 'short'],
                       help='Position sides to test')
    
    parser.add_argument('--max-positions', '-m',
                       type=int,
                       help='Maximum number of positions to open')
    
    parser.add_argument('--no-cleanup',
                       action='store_true',
                       help='Do not cleanup positions after testing')
    
    parser.add_argument('--spot',
                       action='store_true',
                       help='Test spot positions instead of perpetuals')
    
    # Output options
    parser.add_argument('--output', '-f',
                       help='Output results to JSON file')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = PositionTestRunner()
    
    # Handle list command
    if args.list:
        runner.list_configs()
        return
    
    # Create configuration
    overrides = {}
    
    if args.symbols:
        overrides['symbols'] = args.symbols
    if args.sizes:
        overrides['position_sizes'] = args.sizes
    if args.sides:
        overrides['sides'] = args.sides
    if args.max_positions:
        overrides['max_positions'] = args.max_positions
    if args.no_cleanup:
        overrides['cleanup_positions'] = False
    if args.spot:
        overrides['test_perpetuals'] = False
    
    config = runner.create_test_config(args.config, overrides)
    
    # Run tests
    try:
        success = await runner.run_tests(config, args.output)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())