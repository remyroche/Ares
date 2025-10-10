#!/usr/bin/env python3
"""
Position Testing CLI

Command-line interface for testing position management operations.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_position_test_suite import (
    EnhancedPositionTester, PositionTestConfig, 
    PositionSide, OrderType
)

async def run_position_tests(
    symbols: list,
    sizes: list,
    sides: list,
    max_positions: int,
    cleanup: bool,
    test_perpetuals: bool
) -> bool:
    """Run position tests with given configuration."""
    
    # Convert side strings to enums
    side_enums = []
    for side in sides:
        if side.lower() == 'long':
            side_enums.append(PositionSide.LONG)
        elif side.lower() == 'short':
            side_enums.append(PositionSide.SHORT)
        else:
            print(f"⚠️ Unknown side: {side}, skipping")
    
    if not side_enums:
        print("❌ No valid sides specified")
        return False
    
    # Create configuration
    config = PositionTestConfig(
        symbols=symbols,
        position_sizes=sizes,
        sides=side_enums,
        test_perpetuals=test_perpetuals,
        max_positions=max_positions,
        cleanup_positions=cleanup
    )
    
    # Create tester
    tester = EnhancedPositionTester(config)
    
    try:
        # Run tests
        results = await tester.run_all_tests()
        
        # Return success status
        failed_tests = sum(1 for r in results if not r.success)
        return failed_tests == 0
        
    except Exception as e:
        print(f"❌ Position tests failed: {e}")
        return False
    finally:
        await tester.cleanup()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Position Testing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test basic long positions
  python3 test_positions.py --symbols BTCUSDT ETHUSDT --sides long

  # Test both long and short positions
  python3 test_positions.py --symbols BTCUSDT ETHUSDT ADAUSDT --sides long short

  # Test with different sizes
  python3 test_positions.py --symbols BTCUSDT --sizes 0.001 0.01 0.1 --sides long short

  # Test without cleanup (keep positions open)
  python3 test_positions.py --symbols BTCUSDT --sides long --no-cleanup

  # Test spot positions instead of perpetuals
  python3 test_positions.py --symbols BTCUSDT --sides long --spot
        """
    )
    
    parser.add_argument('--symbols', '-s',
                       nargs='+',
                       default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols to test (default: BTCUSDT ETHUSDT)')
    
    parser.add_argument('--sizes', '-z',
                       nargs='+',
                       type=float,
                       default=[0.001, 0.01, 0.1],
                       help='Position sizes to test (default: 0.001 0.01 0.1)')
    
    parser.add_argument('--sides', '-d',
                       nargs='+',
                       choices=['long', 'short'],
                       default=['long', 'short'],
                       help='Position sides to test (default: long short)')
    
    parser.add_argument('--max-positions', '-m',
                       type=int,
                       default=6,
                       help='Maximum number of positions to open (default: 6)')
    
    parser.add_argument('--no-cleanup',
                       action='store_true',
                       help='Do not cleanup positions after testing')
    
    parser.add_argument('--spot',
                       action='store_true',
                       help='Test spot positions instead of perpetuals')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🚀 Position Testing CLI")
    print("=" * 50)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Sizes: {', '.join(map(str, args.sizes))}")
    print(f"Sides: {', '.join(args.sides)}")
    print(f"Max Positions: {args.max_positions}")
    print(f"Cleanup: {not args.no_cleanup}")
    print(f"Type: {'Spot' if args.spot else 'Perpetuals'}")
    print()
    
    # Run tests
    try:
        success = asyncio.run(run_position_tests(
            symbols=args.symbols,
            sizes=args.sizes,
            sides=args.sides,
            max_positions=args.max_positions,
            cleanup=not args.no_cleanup,
            test_perpetuals=not args.spot
        ))
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()