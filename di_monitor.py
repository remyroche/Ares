#!/usr/bin/env python3
"""
Dependency Injection System Monitor

This script monitors the health and functionality of the dependency injection system
across the Ares project, with particular focus on step04 utilities.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

class DIMonitor:
    """Monitor for Dependency Injection system health."""

    def __init__(self):
        self.logger = system_logger.getChild('DIMonitor')
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'components': {},
            'errors': [],
            'warnings': []
        }

    def run_full_diagnosis(self) -> Dict[str, Any]:
        """Run comprehensive DI system diagnosis."""
        self.logger.info("🔍 Starting Dependency Injection System Diagnosis")

        try:
            # Test basic imports
            self._test_basic_imports()

            # Test step04 DI components
            self._test_step04_di()

            # Test utility functions
            self._test_utility_functions()

            # Test DI container functionality
            self._test_di_containers()

            # Generate summary
            self._generate_summary()

        except Exception as e:
            self.logger.error(f"❌ DI diagnosis failed: {e}")
            self.results['errors'].append({
                'component': 'diagnosis_system',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

        return self.results

    def _test_basic_imports(self) -> None:
        """Test basic import functionality."""
        self.logger.info("📦 Testing basic imports...")

        test_imports = [
            ('src.training.steps.model_training.step04_dependency_injection', 'get_step04_utilities'),
            ('src.training.steps.model_training.step04_dependency_injection', 'create_step04_config'),
            ('src.training.steps.model_training.step04_dependency_injection', 'get_step04_container'),
            ('src.core.dependency_injection', 'DependencyContainer'),
            ('src.utils.common_operations', 'safe_float'),
            ('src.utils.math_validation', 'validate_positive'),
        ]

        for module_name, attr_name in test_imports:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                attr = getattr(module, attr_name, None)
                if attr is None:
                    self.results['warnings'].append({
                        'component': 'imports',
                        'message': f"Attribute '{attr_name}' not found in {module_name}"
                    })
                else:
                    self.logger.info(f"✅ {module_name}.{attr_name} - OK")
            except ImportError as e:
                self.results['errors'].append({
                    'component': 'imports',
                    'module': module_name,
                    'attribute': attr_name,
                    'error': str(e)
                })
                self.logger.error(f"❌ Failed to import {module_name}.{attr_name}: {e}")
            except Exception as e:
                self.results['errors'].append({
                    'component': 'imports',
                    'module': module_name,
                    'attribute': attr_name,
                    'error': str(e)
                })
                self.logger.error(f"❌ Unexpected error importing {module_name}.{attr_name}: {e}")

    def _test_step04_di(self) -> None:
        """Test step04 DI components."""
        self.logger.info("🔧 Testing step04 DI components...")

        try:
            from src.training.steps.model_training.step04_dependency_injection import (
                create_step04_config, get_step04_container, get_step04_utilities
            )

            # Test config creation
            config = create_step04_config()
            self.logger.info("✅ Step04 config creation - OK")

            # Test container creation
            container = get_step04_container(config)
            self.logger.info("✅ Step04 container creation - OK")

            # Test utilities creation
            utils = get_step04_utilities()
            self.logger.info("✅ Step04 utilities creation - OK")

            self.results['components']['step04_di'] = {
                'status': 'healthy',
                'config_created': True,
                'container_created': True,
                'utilities_created': True
            }

        except Exception as e:
            self.logger.error(f"❌ Step04 DI test failed: {e}")
            self.results['components']['step04_di'] = {
                'status': 'unhealthy',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _test_utility_functions(self) -> None:
        """Test utility function availability."""
        self.logger.info("🛠️ Testing utility function availability...")

        utility_tests = [
            ('common_operations', 'safe_float', lambda: __import__('src.utils.common_operations', fromlist=['safe_float']).safe_float),
            ('common_operations', 'safe_int', lambda: __import__('src.utils.common_operations', fromlist=['safe_int']).safe_int),
            ('math_validation', 'validate_positive', lambda: __import__('src.utils.math_validation', fromlist=['validate_positive']).validate_positive),
            ('math_validation', 'validate_range', lambda: __import__('src.utils.math_validation', fromlist=['validate_range']).validate_range),
            ('common_utilities', 'create_data_quality_report', lambda: __import__('src.utils.common_utilities', fromlist=['create_data_quality_report']).create_data_quality_report),
        ]

        for utility_type, function_name, import_func in utility_tests:
            try:
                func = import_func()
                if func is None:
                    self.results['warnings'].append({
                        'component': 'utility_functions',
                        'utility_type': utility_type,
                        'function': function_name,
                        'message': f"Function {function_name} is None"
                    })
                else:
                    self.logger.info(f"✅ {utility_type}.{function_name} - OK")

                if utility_type not in self.results['components']:
                    self.results['components'][utility_type] = {}
                self.results['components'][utility_type][function_name] = 'available' if func else 'none'

            except ImportError as e:
                self.logger.error(f"❌ Failed to import {utility_type}.{function_name}: {e}")
                if utility_type not in self.results['components']:
                    self.results['components'][utility_type] = {}
                self.results['components'][utility_type][function_name] = 'import_error'

            except Exception as e:
                self.logger.error(f"❌ Unexpected error with {utility_type}.{function_name}: {e}")
                if utility_type not in self.results['components']:
                    self.results['components'][utility_type] = {}
                self.results['components'][utility_type][function_name] = 'error'

    def _test_di_containers(self) -> None:
        """Test DI container functionality."""
        self.logger.info("📦 Testing DI container functionality...")

        try:
            from src.training.steps.model_training.step04_dependency_injection import get_step04_utilities

            utils = get_step04_utilities()

            # Test utility function access
            test_functions = [
                ('common_operations', 'safe_float'),
                ('math_validation', 'validate_positive'),
                ('common_operations', 'safe_int'),
            ]

            for utility_type, function_name in test_functions:
                try:
                    func = utils.get_function(utility_type, function_name)
                    if func is None:
                        self.results['warnings'].append({
                            'component': 'di_container',
                            'utility_type': utility_type,
                            'function': function_name,
                            'message': f"Function {function_name} returned None from DI container"
                        })
                        self.logger.warning(f"⚠️ {utility_type}.{function_name} returned None from DI")
                    else:
                        self.logger.info(f"✅ DI container: {utility_type}.{function_name} - OK")

                except Exception as e:
                    self.logger.error(f"❌ DI container test failed for {utility_type}.{function_name}: {e}")
                    self.results['errors'].append({
                        'component': 'di_container',
                        'utility_type': utility_type,
                        'function': function_name,
                        'error': str(e)
                    })

        except Exception as e:
            self.logger.error(f"❌ DI container test failed: {e}")
            self.results['errors'].append({
                'component': 'di_container',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def _generate_summary(self) -> None:
        """Generate diagnosis summary."""
        error_count = len(self.results['errors'])
        warning_count = len(self.results['warnings'])

        if error_count == 0 and warning_count == 0:
            self.results['overall_health'] = 'healthy'
            self.logger.info("🎉 DI System Health: HEALTHY")
        elif error_count == 0 and warning_count > 0:
            self.results['overall_health'] = 'warning'
            self.logger.warning(f"⚠️ DI System Health: WARNING ({warning_count} warnings)")
        else:
            self.results['overall_health'] = 'unhealthy'
            self.logger.error(f"❌ DI System Health: UNHEALTHY ({error_count} errors, {warning_count} warnings)")

        self.logger.info("📊 Diagnosis Summary:")
        self.logger.info(f"   Errors: {error_count}")
        self.logger.info(f"   Warnings: {warning_count}")
        self.logger.info(f"   Components tested: {len(self.results['components'])}")

    def print_report(self) -> None:
        """Print detailed diagnosis report."""
        print("\n" + "="*80)
        print("DEPENDENCY INJECTION SYSTEM DIAGNOSIS REPORT")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Health: {self.results['overall_health'].upper()}")
        print()

        if self.results['errors']:
            print("❌ ERRORS:")
            for error in self.results['errors']:
                print(f"   - {error.get('component', 'unknown')}: {error.get('error', 'Unknown error')}")
                if 'module' in error:
                    print(f"     Module: {error['module']}.{error.get('attribute', '')}")
            print()

        if self.results['warnings']:
            print("⚠️ WARNINGS:")
            for warning in self.results['warnings']:
                print(f"   - {warning.get('component', 'unknown')}: {warning.get('message', 'Unknown warning')}")
            print()

        print("📦 COMPONENT STATUS:")
        for component, status in self.results['components'].items():
            print(f"   {component}:")
            if isinstance(status, dict):
                for key, value in status.items():
                    status_icon = "✅" if value in ['available', 'healthy', True] else "❌" if value in ['none', 'unhealthy', False] else "⚠️"
                    print(f"     - {key}: {status_icon} {value}")
            else:
                print(f"     - Status: {status}")
        print()

def main():
    """Main entry point for DI monitoring."""
    print("🔍 Starting Dependency Injection System Monitor...")

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    monitor = DIMonitor()
    results = monitor.run_full_diagnosis()
    monitor.print_report()

    # Save results to file
    import json
    output_file = Path(project_root) / "di_diagnosis_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"📄 Detailed report saved to: {output_file}")

    return 0 if results['overall_health'] == 'healthy' else 1

if __name__ == '__main__':
    sys.exit(main())
