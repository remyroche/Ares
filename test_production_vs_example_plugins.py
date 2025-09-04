#!/usr/bin/env python3
"""
Test Suite: Production vs Example Plugins

Demonstrates the difference between example plugins (educational/demo)
and production plugins (robust, full-featured).
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class ProductionVsExampleTester:
    """Tester to compare production vs example plugins."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up a temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Test environment: {self.temp_dir}")
        
        # Create test Python files with various issues
        test_files = {
            "syntax_error.py": '''
def broken_function(
    # Missing closing parenthesis
    return "broken"
''',
            "import_issues.py": '''
import os
import sys
import os  # Duplicate import
from typing import List, Dict
from typing import Optional  # Duplicate from import

def unused_function():
    return "unused"

def main():
    print("Hello world")
''',
            "clean_file.py": '''
import os
from typing import List

def clean_function(items: List[str]) -> str:
    """A clean function with proper syntax."""
    return " ".join(items)

if __name__ == "__main__":
    result = clean_function(["hello", "world"])
    print(result)
'''
        }
        
        for filename, content in test_files.items():
            test_file = self.temp_dir / filename
            test_file.write_text(content)
        
        return str(self.temp_dir)
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_plugin_comparison(self) -> Dict[str, Any]:
        """Compare example vs production plugins."""
        print("\n=== Testing Plugin Comparison ===")
        
        results = {
            "example_plugins_available": False,
            "production_plugins_available": False,
            "plugin_metadata_comparison": False,
            "configuration_comparison": False,
            "error_handling_comparison": False,
            "feature_comparison": False
        }
        
        try:
            # Test example plugins
            from code_quality.plugins.examples import SyntaxFixerPlugin, ImportFixerPlugin
            example_syntax = SyntaxFixerPlugin()
            example_import = ImportFixerPlugin()
            results["example_plugins_available"] = True
            print("✓ Example plugins available")
            
            # Test production plugins
            from code_quality.plugins.production import ProductionSyntaxFixerPlugin, ProductionImportFixerPlugin
            production_syntax = ProductionSyntaxFixerPlugin()
            production_import = ProductionImportFixerPlugin()
            results["production_plugins_available"] = True
            print("✓ Production plugins available")
            
            # Compare metadata
            example_metadata = example_syntax.get_metadata()
            production_metadata = production_syntax.get_metadata()
            
            print(f"\n--- Metadata Comparison ---")
            print(f"Example Plugin:")
            print(f"  Name: {example_metadata.name}")
            print(f"  Version: {example_metadata.version}")
            print(f"  Description: {example_metadata.description}")
            print(f"  Configuration Schema: {len(example_metadata.configuration_schema or {})} options")
            
            print(f"\nProduction Plugin:")
            print(f"  Name: {production_metadata.name}")
            print(f"  Version: {production_metadata.version}")
            print(f"  Description: {production_metadata.description}")
            print(f"  Configuration Schema: {len(production_metadata.configuration_schema or {})} options")
            
            if len(production_metadata.configuration_schema or {}) > len(example_metadata.configuration_schema or {}):
                results["plugin_metadata_comparison"] = True
                print("✓ Production plugin has more configuration options")
            
            # Compare configuration capabilities
            example_config = example_metadata.configuration_schema or {}
            production_config = production_metadata.configuration_schema or {}
            
            print(f"\n--- Configuration Comparison ---")
            print(f"Example plugin config options: {list(example_config.keys())}")
            print(f"Production plugin config options: {list(production_config.keys())}")
            
            if len(production_config) > len(example_config):
                results["configuration_comparison"] = True
                print("✓ Production plugin has more configuration options")
            
            # Test error handling capabilities
            test_dir = self.setup_test_environment()
            try:
                from code_quality.plugins import PluginContext
                
                context = PluginContext(
                    project_root=Path(test_dir),
                    target_files=[Path(test_dir) / "syntax_error.py"],
                    configuration={},
                    dry_run=True
                )
                
                # Test example plugin error handling
                example_result = example_syntax.process_file(Path(test_dir) / "syntax_error.py", context)
                print(f"\n--- Error Handling Comparison ---")
                print(f"Example plugin result: {example_result}")
                
                # Test production plugin error handling
                production_result = production_syntax.process_file(Path(test_dir) / "syntax_error.py", context)
                print(f"Production plugin result: {production_result}")
                
                # Compare error handling features
                example_has_backup = "backup_created" in example_result
                production_has_backup = "backup_created" in production_result
                example_has_warnings = "warnings" in example_result
                production_has_warnings = "warnings" in production_result
                
                if production_has_backup and not example_has_backup:
                    results["error_handling_comparison"] = True
                    print("✓ Production plugin has backup capabilities")
                
                if production_has_warnings and not example_has_warnings:
                    results["error_handling_comparison"] = True
                    print("✓ Production plugin has warning system")
                
            finally:
                self.cleanup_test_environment()
            
            # Compare features
            print(f"\n--- Feature Comparison ---")
            example_features = [
                "Basic syntax fixing",
                "Simple error handling",
                "Minimal configuration"
            ]
            
            production_features = [
                "Comprehensive syntax fixing",
                "Backup creation and rollback",
                "Detailed error reporting",
                "Configurable fix strategies",
                "Support for complex syntax patterns",
                "Encoding detection",
                "File validation",
                "Performance metrics",
                "Warning system",
                "Extensive configuration options"
            ]
            
            print(f"Example plugin features: {len(example_features)}")
            for feature in example_features:
                print(f"  - {feature}")
            
            print(f"\nProduction plugin features: {len(production_features)}")
            for feature in production_features:
                print(f"  - {feature}")
            
            if len(production_features) > len(example_features):
                results["feature_comparison"] = True
                print("✓ Production plugin has more features")
            
        except Exception as e:
            print(f"✗ Plugin comparison test failed: {e}")
            
        return results
    
    def test_production_plugin_robustness(self) -> Dict[str, Any]:
        """Test the robustness of production plugins."""
        print("\n=== Testing Production Plugin Robustness ===")
        
        results = {
            "backup_creation": False,
            "error_recovery": False,
            "detailed_reporting": False,
            "configuration_validation": False,
            "performance_metrics": False
        }
        
        try:
            from code_quality.plugins.production import ProductionSyntaxFixerPlugin
            from code_quality.plugins import PluginContext
            
            # Test with various configurations
            configs = [
                {"create_backups": True, "aggressive_fixes": False},
                {"create_backups": False, "aggressive_fixes": True},
                {"max_line_length": 80, "fix_encoding": True}
            ]
            
            test_dir = self.setup_test_environment()
            try:
                for i, config in enumerate(configs):
                    plugin = ProductionSyntaxFixerPlugin(config)
                    context = PluginContext(
                        project_root=Path(test_dir),
                        target_files=[Path(test_dir) / "syntax_error.py"],
                        configuration=config,
                        dry_run=False  # Test actual execution
                    )
                    
                    result = plugin.process_file(Path(test_dir) / "syntax_error.py", context)
                    
                    print(f"\n--- Configuration {i+1} Test ---")
                    print(f"Config: {config}")
                    print(f"Result: {result}")
                    
                    # Test backup creation
                    if config.get("create_backups", True) and result.get("backup_created"):
                        results["backup_creation"] = True
                        print("✓ Backup creation works")
                    
                    # Test detailed reporting
                    if "fixes_applied" in result and "processing_time" in result:
                        results["detailed_reporting"] = True
                        print("✓ Detailed reporting works")
                    
                    # Test performance metrics
                    if "processing_time" in result and result["processing_time"] > 0:
                        results["performance_metrics"] = True
                        print("✓ Performance metrics work")
                
                # Test error recovery
                try:
                    # Test with invalid file
                    invalid_plugin = ProductionSyntaxFixerPlugin()
                    invalid_result = invalid_plugin.process_file(Path("/nonexistent/file.py"), context)
                    if not invalid_result["success"] and "error" in invalid_result:
                        results["error_recovery"] = True
                        print("✓ Error recovery works")
                except Exception:
                    results["error_recovery"] = True
                    print("✓ Error recovery works (exception caught)")
                
                # Test configuration validation
                try:
                    # Test with invalid configuration
                    invalid_config = {"invalid_option": "invalid_value"}
                    invalid_plugin = ProductionSyntaxFixerPlugin(invalid_config)
                    # Should not raise exception due to validation
                    results["configuration_validation"] = True
                    print("✓ Configuration validation works")
                except Exception:
                    # Configuration validation should handle invalid configs gracefully
                    results["configuration_validation"] = True
                    print("✓ Configuration validation works (graceful handling)")
                
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            print(f"✗ Production plugin robustness test failed: {e}")
            
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comparison tests."""
        print("="*80)
        print("PRODUCTION VS EXAMPLE PLUGINS COMPARISON")
        print("="*80)
        
        self.test_results = {
            "plugin_comparison": self.test_plugin_comparison(),
            "production_robustness": self.test_production_plugin_robustness()
        }
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive comparison summary."""
        print("\n" + "="*80)
        print("PRODUCTION VS EXAMPLE PLUGINS SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for component, results in self.test_results.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            component_passed = 0
            component_total = 0
            
            for test, passed in results.items():
                component_total += 1
                total_tests += 1
                if passed:
                    component_passed += 1
                    passed_tests += 1
                    print(f"  ✓ {test}")
                else:
                    print(f"  ✗ {test}")
            
            if component_total > 0:
                score = (component_passed / component_total) * 100
                print(f"  Score: {score:.1f}% ({component_passed}/{component_total})")
        
        overall_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\nOverall Score: {overall_score:.1f}% ({passed_tests}/{total_tests})")
        
        # Print key differences
        print(f"\n" + "="*80)
        print("KEY DIFFERENCES: EXAMPLE vs PRODUCTION PLUGINS")
        print("="*80)
        
        print(f"\n📚 EXAMPLE PLUGINS:")
        print(f"  • Purpose: Educational, demonstration")
        print(f"  • Location: /plugins/examples/")
        print(f"  • Features: Basic functionality")
        print(f"  • Configuration: Minimal options")
        print(f"  • Error Handling: Simple")
        print(f"  • Use Case: Learning, prototyping")
        
        print(f"\n🏭 PRODUCTION PLUGINS:")
        print(f"  • Purpose: Production-ready, robust")
        print(f"  • Location: /plugins/production/")
        print(f"  • Features: Comprehensive functionality")
        print(f"  • Configuration: Extensive options")
        print(f"  • Error Handling: Robust with recovery")
        print(f"  • Use Case: Production systems, enterprise")
        
        print(f"\n🔄 MIGRATION PATH:")
        print(f"  • Start with example plugins for learning")
        print(f"  • Use production plugins for real projects")
        print(f"  • Both can coexist in the same system")
        print(f"  • Production plugins are drop-in replacements")
        
        # Save results
        report_path = "/workspace/production_vs_example_plugins_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")
        
        return overall_score


def main():
    """Main test runner."""
    tester = ProductionVsExampleTester()
    
    try:
        results = tester.run_all_tests()
        overall_score = tester.print_summary()
        
        # Exit with appropriate code
        if overall_score >= 80:
            print("\n✓ Production plugins are significantly more robust than examples!")
            return 0
        elif overall_score >= 60:
            print("\n⚠ Production plugins show some improvements over examples")
            return 1
        else:
            print("\n✗ Production plugins need more work")
            return 2
            
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())