#!/usr/bin/env python3
"""
Enhanced Exchange Interface Test Runner

A more user-friendly CLI tool that supports configuration files and presets.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from exchange_interface_test_suite import ExchangeInterfaceTestSuite, TestConfig

class TestRunner:
    """Enhanced test runner with configuration support."""
    
    def __init__(self):
        self.config_file = Path("test_config.json")
        self.configs = {}
        self.load_configs()
    
    def load_configs(self) -> None:
        """Load test configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.configs = data.get('test_configs', {})
                    self.default_config = data.get('default_config', 'simulated')
            except Exception as e:
                print(f"⚠️ Warning: Could not load config file: {e}")
                self.configs = {}
                self.default_config = 'simulated'
        else:
            print("ℹ️ No config file found, using defaults")
            self.configs = {}
            self.default_config = 'simulated'
    
    def list_configs(self) -> None:
        """List available test configurations."""
        print("📋 Available Test Configurations:")
        print("=" * 50)
        
        if not self.configs:
            print("No configurations found in test_config.json")
            return
        
        for name, config in self.configs.items():
            print(f"\n🔧 {name}:")
            print(f"   Exchange: {config.get('exchange_type', 'N/A')}")
            print(f"   Symbol: {config.get('test_symbol', 'N/A')}")
            print(f"   Interval: {config.get('test_interval', 'N/A')}")
            print(f"   Testnet: {config.get('testnet', 'N/A')}")
            print(f"   Operations: {', '.join(config.get('test_operations', []))}")
    
    def create_test_config(self, config_name: Optional[str] = None, 
                          overrides: Optional[Dict[str, Any]] = None) -> TestConfig:
        """Create a test configuration from file or defaults."""
        
        # Start with default config
        if config_name and config_name in self.configs:
            config_data = self.configs[config_name].copy()
        elif self.default_config in self.configs:
            config_data = self.configs[self.default_config].copy()
        else:
            config_data = {
                "exchange_type": "simulated",
                "test_symbol": "BTCUSDT",
                "test_interval": "1m",
                "test_quantity": 0.001,
                "testnet": True,
                "verbose": True,
                "test_operations": ["connection", "klines", "balance", "ticker"]
            }
        
        # Apply overrides
        if overrides:
            config_data.update(overrides)
        
        # Create TestConfig object
        return TestConfig(
            exchange_type=config_data.get('exchange_type', 'simulated'),
            test_symbol=config_data.get('test_symbol', 'BTCUSDT'),
            test_interval=config_data.get('test_interval', '1m'),
            test_quantity=config_data.get('test_quantity', 0.001),
            api_key=config_data.get('api_key'),
            api_secret=config_data.get('api_secret'),
            testnet=config_data.get('testnet', True),
            verbose=config_data.get('verbose', True),
            timeout=config_data.get('timeout', 30),
            test_operations=config_data.get('test_operations', [])
        )
    
    async def run_tests(self, config: TestConfig, output_file: Optional[str] = None) -> bool:
        """Run the test suite with the given configuration."""
        print(f"🚀 Running Exchange Interface Tests")
        print(f"📊 Exchange: {config.exchange_type}")
        print(f"💰 Symbol: {config.test_symbol}")
        print(f"⏱️ Interval: {config.test_interval}")
        print(f"🔧 Operations: {', '.join(config.test_operations)}")
        print("=" * 60)
        
        test_suite = ExchangeInterfaceTestSuite(config)
        
        try:
            results = await test_suite.run_all_tests()
            
            # Save results to file if requested
            if output_file:
                self.save_results(results, output_file, config)
            
            # Return success status
            return results.failed_tests == 0
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            return False
        finally:
            await test_suite.cleanup()
    
    def save_results(self, results, output_file: str, config: TestConfig) -> None:
        """Save test results to a JSON file."""
        try:
            output_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "config": {
                    "exchange_type": config.exchange_type,
                    "test_symbol": config.test_symbol,
                    "test_interval": config.test_interval,
                    "test_operations": config.test_operations
                },
                "results": {
                    "total_tests": results.total_tests,
                    "passed_tests": results.passed_tests,
                    "failed_tests": results.failed_tests,
                    "total_duration": results.total_duration,
                    "success_rate": (results.passed_tests / results.total_tests * 100) if results.total_tests > 0 else 0
                },
                "test_details": [
                    {
                        "operation": r.operation,
                        "success": r.success,
                        "duration": r.duration,
                        "error": r.error,
                        "data": r.data
                    } for r in results.results
                ],
                "errors": results.errors,
                "warnings": results.warnings
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"📄 Results saved to {output_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save results to {output_file}: {e}")

def main():
    """Main entry point for the enhanced test runner."""
    parser = argparse.ArgumentParser(
        description="Enhanced Exchange Interface Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configurations
  python run_exchange_tests.py --list

  # Run with a specific configuration
  python run_exchange_tests.py --config binance_testnet

  # Run with overrides
  python run_exchange_tests.py --config simulated --symbol ETHUSDT --interval 5m

  # Run with custom settings
  python run_exchange_tests.py --exchange binance --symbol BTCUSDT --operations klines,balance

  # Save results to file
  python run_exchange_tests.py --config binance_testnet --output results.json
        """
    )
    
    # Configuration options
    parser.add_argument('--config', '-c',
                       help='Use a predefined configuration from test_config.json')
    
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='List available configurations')
    
    # Exchange options
    parser.add_argument('--exchange', '-e',
                       choices=['simulated', 'binance', 'coinbase', 'kraken', 'bybit'],
                       help='Exchange type to test')
    
    parser.add_argument('--symbol', '-s',
                       help='Trading symbol to test')
    
    parser.add_argument('--interval', '-i',
                       help='Kline interval to test')
    
    parser.add_argument('--operations', '-o',
                       help='Comma-separated list of operations to test')
    
    # API options
    parser.add_argument('--api-key',
                       help='API key for live exchange testing')
    
    parser.add_argument('--api-secret',
                       help='API secret for live exchange testing')
    
    parser.add_argument('--live',
                       action='store_true',
                       help='Use live exchange (default: testnet)')
    
    # Output options
    parser.add_argument('--output', '-f',
                       help='Output results to JSON file')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Handle list command
    if args.list:
        runner.list_configs()
        return
    
    # Create configuration
    overrides = {}
    
    if args.exchange:
        overrides['exchange_type'] = args.exchange
    if args.symbol:
        overrides['test_symbol'] = args.symbol
    if args.interval:
        overrides['test_interval'] = args.interval
    if args.operations:
        overrides['test_operations'] = args.operations.split(',')
    if args.api_key:
        overrides['api_key'] = args.api_key
    if args.api_secret:
        overrides['api_secret'] = args.api_secret
    if args.live:
        overrides['testnet'] = False
    if args.verbose:
        overrides['verbose'] = True
    
    config = runner.create_test_config(args.config, overrides)
    
    # Run tests
    async def run():
        success = await runner.run_tests(config, args.output)
        return success
    
    try:
        success = asyncio.run(run())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()