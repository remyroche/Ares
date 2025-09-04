#!/usr/bin/env python3
"""
Test Suite for Plugin System

Tests the plugin architecture, base class improvements, and integration.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class PluginSystemTester:
    """Comprehensive tester for the plugin system."""
    
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
    
    def test_plugin_registry(self) -> Dict[str, Any]:
        """Test the plugin registry functionality."""
        print("\n=== Testing Plugin Registry ===")
        
        results = {
            "imports_work": False,
            "registry_creation": False,
            "plugin_discovery": False,
            "plugin_registration": False,
            "plugin_retrieval": False
        }
        
        try:
            # Test imports
            from code_quality.plugins import PluginRegistry, BasePlugin
            from code_quality.plugins.examples import SyntaxFixerPlugin
            results["imports_work"] = True
            print("✓ Plugin imports successful")
            
            # Test registry creation
            registry = PluginRegistry()
            results["registry_creation"] = True
            print("✓ Plugin registry creation successful")
            
            # Test plugin registration
            registry.register_plugin(SyntaxFixerPlugin)
            results["plugin_registration"] = True
            print("✓ Plugin registration successful")
            
            # Test plugin retrieval
            plugin = registry.get_plugin("syntax_fixer")
            if plugin is not None:
                results["plugin_retrieval"] = True
                print("✓ Plugin retrieval successful")
            
            # Test plugin discovery
            examples_dir = Path("/workspace/code_quality/plugins/examples")
            if examples_dir.exists():
                discovered = registry.discover_plugins(examples_dir)
                if discovered > 0:
                    results["plugin_discovery"] = True
                    print(f"✓ Plugin discovery successful: {discovered} plugins found")
            
        except Exception as e:
            print(f"✗ Plugin registry test failed: {e}")
            
        return results
    
    def test_plugin_manager(self) -> Dict[str, Any]:
        """Test the plugin manager functionality."""
        print("\n=== Testing Plugin Manager ===")
        
        results = {
            "imports_work": False,
            "manager_creation": False,
            "plugin_execution": False,
            "parallel_execution": False,
            "error_handling": False
        }
        
        try:
            # Test imports
            from code_quality.plugins import PluginManager, PluginRegistry, PluginContext
            from code_quality.plugins.examples import SyntaxFixerPlugin
            results["imports_work"] = True
            print("✓ Plugin manager imports successful")
            
            # Test manager creation
            registry = PluginRegistry()
            registry.register_plugin(SyntaxFixerPlugin)
            manager = PluginManager(registry)
            results["manager_creation"] = True
            print("✓ Plugin manager creation successful")
            
            # Test plugin execution
            test_dir = self.setup_test_environment()
            try:
                context = PluginContext(
                    project_root=Path(test_dir),
                    target_files=[Path(test_dir) / "syntax_error.py"],
                    configuration={},
                    dry_run=True
                )
                
                result = manager.execute_plugin("syntax_fixer", context)
                if result is not None:
                    results["plugin_execution"] = True
                    print("✓ Plugin execution successful")
                
            except Exception as e:
                print(f"✗ Plugin execution failed: {e}")
            finally:
                self.cleanup_test_environment()
            
            # Test error handling
            try:
                manager.execute_plugin("non_existent_plugin", context)
            except Exception:
                results["error_handling"] = True
                print("✓ Error handling works correctly")
            
        except Exception as e:
            print(f"✗ Plugin manager test failed: {e}")
            
        return results
    
    def test_base_pipeline_enhancements(self) -> Dict[str, Any]:
        """Test the enhanced base pipeline class."""
        print("\n=== Testing Enhanced Base Pipeline ===")
        
        results = {
            "imports_work": False,
            "pipeline_creation": False,
            "plugin_integration": False,
            "configuration_management": False,
            "metrics_collection": False
        }
        
        try:
            # Test imports
            from code_quality.pipelines.base_pipeline import BasePipeline, PipelineConfig
            from code_quality.plugins import PluginCategory, PluginPriority
            results["imports_work"] = True
            print("✓ Enhanced base pipeline imports successful")
            
            # Test pipeline creation with configuration
            test_dir = self.setup_test_environment()
            try:
                config = PipelineConfig(
                    project_root=Path(test_dir),
                    output_dir=Path(test_dir) / "output",
                    parallel_execution=True,
                    max_workers=2,
                    log_level="INFO",
                    plugin_categories=[PluginCategory.SYNTAX],
                    plugin_priorities=[PluginPriority.HIGH]
                )
                
                pipeline = BasePipeline(project_root=test_dir, config=config)
                results["pipeline_creation"] = True
                print("✓ Enhanced pipeline creation successful")
                
                # Test plugin integration
                available_plugins = pipeline.get_available_plugins()
                if available_plugins:
                    results["plugin_integration"] = True
                    print(f"✓ Plugin integration successful: {len(available_plugins)} plugins available")
                
                # Test configuration management
                config_errors = config.validate()
                if not config_errors:
                    results["configuration_management"] = True
                    print("✓ Configuration management successful")
                
                # Test metrics collection
                metrics = pipeline.get_metrics()
                if isinstance(metrics, dict) and "execution_count" in metrics:
                    results["metrics_collection"] = True
                    print("✓ Metrics collection successful")
                
            except Exception as e:
                print(f"✗ Enhanced pipeline test failed: {e}")
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            print(f"✗ Enhanced base pipeline test failed: {e}")
            
        return results
    
    def test_plugin_examples(self) -> Dict[str, Any]:
        """Test the example plugins."""
        print("\n=== Testing Example Plugins ===")
        
        results = {
            "syntax_plugin": False,
            "import_plugin": False,
            "linter_plugin": False,
            "security_plugin": False,
            "plugin_metadata": False
        }
        
        try:
            # Test imports
            from code_quality.plugins.examples import (
                SyntaxFixerPlugin, ImportFixerPlugin, 
                LinterPlugin, SecurityScannerPlugin
            )
            print("✓ Example plugin imports successful")
            
            # Test syntax plugin
            syntax_plugin = SyntaxFixerPlugin()
            metadata = syntax_plugin.get_metadata()
            if metadata.name == "syntax_fixer":
                results["syntax_plugin"] = True
                print("✓ Syntax plugin creation successful")
            
            # Test import plugin
            import_plugin = ImportFixerPlugin()
            metadata = import_plugin.get_metadata()
            if metadata.name == "import_fixer":
                results["import_plugin"] = True
                print("✓ Import plugin creation successful")
            
            # Test linter plugin
            linter_plugin = LinterPlugin()
            metadata = linter_plugin.get_metadata()
            if metadata.name == "linter":
                results["linter_plugin"] = True
                print("✓ Linter plugin creation successful")
            
            # Test security plugin
            security_plugin = SecurityScannerPlugin()
            metadata = security_plugin.get_metadata()
            if metadata.name == "security_scanner":
                results["security_plugin"] = True
                print("✓ Security plugin creation successful")
            
            # Test plugin metadata
            if all([
                syntax_plugin.get_metadata().version,
                import_plugin.get_metadata().description,
                linter_plugin.get_metadata().category,
                security_plugin.get_metadata().priority
            ]):
                results["plugin_metadata"] = True
                print("✓ Plugin metadata validation successful")
            
        except Exception as e:
            print(f"✗ Example plugins test failed: {e}")
            
        return results
    
    def test_integration(self) -> Dict[str, Any]:
        """Test integration between components."""
        print("\n=== Testing Integration ===")
        
        results = {
            "pipeline_plugin_execution": False,
            "error_propagation": False,
            "metrics_integration": False,
            "configuration_flow": False
        }
        
        try:
            from code_quality.pipelines.base_pipeline import BasePipeline, PipelineConfig
            from code_quality.plugins import PluginCategory, PluginPriority
            
            # Test pipeline with plugin execution
            test_dir = self.setup_test_environment()
            try:
                config = PipelineConfig(
                    project_root=Path(test_dir),
                    output_dir=Path(test_dir) / "output",
                    plugin_categories=[PluginCategory.SYNTAX],
                    dry_run=True
                )
                
                pipeline = BasePipeline(project_root=test_dir, config=config)
                
                # Test plugin execution through pipeline
                result = pipeline.execute_plugins()
                if isinstance(result, dict) and "pipeline_info" in result:
                    results["pipeline_plugin_execution"] = True
                    print("✓ Pipeline plugin execution successful")
                
                # Test metrics integration
                metrics = pipeline.get_metrics()
                if "plugin_metrics" in metrics:
                    results["metrics_integration"] = True
                    print("✓ Metrics integration successful")
                
                # Test configuration flow
                if pipeline.config.plugin_categories == [PluginCategory.SYNTAX]:
                    results["configuration_flow"] = True
                    print("✓ Configuration flow successful")
                
            except Exception as e:
                print(f"✗ Integration test failed: {e}")
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all plugin system tests."""
        print("="*80)
        print("PLUGIN SYSTEM TESTING")
        print("="*80)
        
        self.test_results = {
            "plugin_registry": self.test_plugin_registry(),
            "plugin_manager": self.test_plugin_manager(),
            "base_pipeline_enhancements": self.test_base_pipeline_enhancements(),
            "plugin_examples": self.test_plugin_examples(),
            "integration": self.test_integration()
        }
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive test summary."""
        print("\n" + "="*80)
        print("PLUGIN SYSTEM TEST SUMMARY")
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
        
        # Save results
        report_path = "/workspace/plugin_system_test_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")
        
        return overall_score


def main():
    """Main test runner."""
    tester = PluginSystemTester()
    
    try:
        results = tester.run_all_tests()
        overall_score = tester.print_summary()
        
        # Exit with appropriate code
        if overall_score >= 80:
            print("\n✓ Plugin system is working well!")
            return 0
        elif overall_score >= 60:
            print("\n⚠ Plugin system has some issues but is mostly functional")
            return 1
        else:
            print("\n✗ Plugin system has significant issues")
            return 2
            
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())