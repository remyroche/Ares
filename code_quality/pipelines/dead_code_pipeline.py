#!/usr/bin/env python3
"""
Dead Code Analysis Pipeline

Specialized pipeline for dead code detection and analysis including:
- Unused imports detection
- Dead function detection
- Unreachable code detection
- Unused variables detection
- Dead code removal automation
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import dead code analyzers (ONLY dead code-related)
from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
from analyzers.enhanced_import_analysis import EnhancedImportAnalyzer

# Import interaction mapping components for enhanced analysis
from mappers.map_code_interactions import CodeInteractionMapper
from analyzers.call_graph_analyzer import CallGraphAnalyzer
from analyzers.dependency_analyzer import DependencyAnalyzer

# Note: AutoFixDeadCode and DeadCodeAnalyzer were removed as they were redundant
# The enhanced_dead_code_analyzer provides all necessary functionality

# Import core components
from core.config import AnalysisConfig

# Import standardized base pipeline
from pipelines.base_pipeline import BasePipeline


class InteractionAwareDeadCodeAnalyzer:
    """Enhanced dead code analyzer that uses interaction mapping data to reduce false positives."""

    def __init__(self, interaction_data: Optional[Dict[str, Any]] = None):
        self.interaction_data = interaction_data or {}
        self.used_functions = set()
        self.used_classes = set()
        self.used_methods = set()
        self.call_graph = defaultdict(set)
        self.import_graph = defaultdict(set)
        self.entry_points = set()

        # Build usage graphs from interaction data if available
        if interaction_data:
            self._build_usage_graphs()

    def _build_usage_graphs(self):
        """Build comprehensive usage graphs from interaction mapping data."""
        print("🔗 Building usage graphs from interaction mapping data...")

        interactions = self.interaction_data.get('results', {}).get('interactions', [])

        for interaction in interactions:
            interaction_type = interaction.get('type', '')
            source = interaction.get('source', '')
            target = interaction.get('target', '')
            source_file = interaction.get('source_file', '')

            if interaction_type == 'function_call':
                self.call_graph[source].add(target)
                self.used_functions.add(target)

                # Track class methods
                if '.' in target:
                    class_name = target.split('.')[0]
                    self.used_classes.add(class_name)
                    self.used_methods.add(target)

            elif interaction_type == 'class_instantiation':
                self.used_classes.add(target)
                self.call_graph[source].add(target)

            elif interaction_type == 'import':
                if source_file and target:
                    self.import_graph[source_file].add(target)
                    if '.' in target:
                        class_name = target.split('.')[-1]
                        self.used_classes.add(class_name)
                    else:
                        self.used_functions.add(target)

        # Extract entry points from call graph analysis
        call_graph_results = self.interaction_data.get('results', {}).get('call_graph_analysis', {})
        if call_graph_results:
            for func in call_graph_results.get('entry_points', []):
                self.entry_points.add(func)

        print(f"✅ Built enhanced usage graphs:")
        print(f"   📊 {len(self.used_functions)} used functions")
        print(f"   📊 {len(self.used_classes)} used classes")
        print(f"   📊 {len(self.used_methods)} used methods")
        print(f"   📊 {len(self.call_graph)} call graph nodes")
        print(f"   📊 {len(self.entry_points)} identified entry points")

    def is_function_used(self, func_name: str, file_path: str = None) -> tuple[bool, str]:
        """Check if a function is used based on interaction data."""
        if func_name in self.used_functions:
            return True, "found_in_interactions"

        if func_name in self.entry_points:
            return True, "entry_point"

        # Check if it's called by any entry point (transitive usage)
        if self._is_reachable_from_entry(func_name):
            return True, "reachable_from_entry"

        return False, "not_found"

    def is_class_used(self, class_name: str, file_path: str = None) -> tuple[bool, str]:
        """Check if a class is used based on interaction data."""
        if class_name in self.used_classes:
            return True, "found_in_interactions"

        # Check if any method of this class is used
        for method in self.used_methods:
            if method.startswith(f"{class_name}."):
                return True, "method_used"

        return False, "not_found"

    def _is_reachable_from_entry(self, func_name: str) -> bool:
        """Check if a function is reachable from any entry point."""
        visited = set()
        to_visit = list(self.entry_points)

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            if current == func_name:
                return True

            # Add all functions called by current function
            to_visit.extend(self.call_graph.get(current, []))

        return False

    def enhance_dead_code_report(self, dead_code_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance dead code analysis results with interaction mapping insights."""
        print("🔍 Enhancing dead code analysis with interaction mapping insights...")

        enhanced_results = dead_code_results.copy()

        # Validate potentially unused functions
        validated_functions = []
        for func in dead_code_results.get('unused_functions', []):
            func_name = func.get('name', '')
            file_path = func.get('file', '')

            is_used, reason = self.is_function_used(func_name, file_path)
            if is_used:
                # Function is actually used, remove from unused list
                print(f"✅ Function '{func_name}' in {file_path} is actually used: {reason}")
            else:
                # Function appears unused, keep in list with additional confidence
                func['confidence'] = 'high'  # Interaction data confirms it's unused
                func['validation_reason'] = reason
                validated_functions.append(func)

        # Validate potentially unused classes
        validated_classes = []
        for cls in dead_code_results.get('unused_classes', []):
            class_name = cls.get('name', '')
            file_path = cls.get('file', '')

            is_used, reason = self.is_class_used(class_name, file_path)
            if is_used:
                print(f"✅ Class '{class_name}' in {file_path} is actually used: {reason}")
            else:
                cls['confidence'] = 'high'
                cls['validation_reason'] = reason
                validated_classes.append(cls)

        # Update results with validated data
        enhanced_results['unused_functions'] = validated_functions
        enhanced_results['unused_classes'] = validated_classes
        enhanced_results['statistics']['false_positives_removed'] = (
            len(dead_code_results.get('unused_functions', [])) - len(validated_functions) +
            len(dead_code_results.get('unused_classes', [])) - len(validated_classes)
        )

        print(f"📊 Enhanced analysis results:")
        print(f"   ✅ False positives removed: {enhanced_results['statistics']['false_positives_removed']}")
        print(f"   🔴 Confirmed unused functions: {len(validated_functions)}")
        print(f"   🔴 Confirmed unused classes: {len(validated_classes)}")

        return enhanced_results


class DeadCodePipeline(BasePipeline):
    """Specialized pipeline for dead code analysis with standardized initialization."""

    def __init__(self, project_root: str = None, enable_plugins: bool = True,
                 use_interaction_mapping: bool = True):
        # Use standardized initialization from base class
        super().__init__(project_root=project_root, enable_plugins=enable_plugins,
                        pipeline_name="dead_code")

        # Setup pipeline-specific paths
        self.setup_pipeline_paths()

        # Initialize analyzers with standardized config
        self.config = AnalysisConfig()
        self.dead_code_analyzer = EnhancedDeadCodeAnalyzer(self.config)
        self.undefined_names_analyzer = UndefinedNamesAnalyzer(self.config)
        self.import_analyzer = EnhancedImportAnalyzer(None)  # Use default config

        # Initialize interaction mapping components for enhanced analysis
        self.use_interaction_mapping = use_interaction_mapping
        self.interaction_mapper = None
        self.call_graph_analyzer = None
        self.dependency_analyzer = None
        self.interaction_data = {}
        self.interaction_aware_analyzer = None

        if self.use_interaction_mapping:
            self._initialize_interaction_components()

        # Initialize plugin system with standardized registration
        if self.enable_plugins:
            self._register_dead_code_plugins()

    def _initialize_interaction_components(self):
        """Initialize interaction mapping components for enhanced dead code analysis."""
        try:
            print("🔗 Initializing interaction mapping components for enhanced analysis...")

            # Initialize interaction mappers
            self.interaction_mapper = CodeInteractionMapper(str(self.project_root))

            # Initialize analyzers
            self.call_graph_analyzer = CallGraphAnalyzer(self.config)
            self.dependency_analyzer = DependencyAnalyzer(self.config)

            print("✅ Interaction mapping components initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize interaction mapping components: {e}")
            self.use_interaction_mapping = False

    def _run_interaction_mapping_analysis(self) -> Dict[str, Any]:
        """Run interaction mapping analysis to gather usage data."""
        if not self.use_interaction_mapping or not self.interaction_mapper:
            return {}

        print("\n🔗 Running interaction mapping analysis for dead code enhancement...")

        try:
            # Run basic interaction mapping
            interaction_results = self.interaction_mapper.map_interactions(str(self.project_root))

            # Run call graph analysis
            call_graph_results = self.call_graph_analyzer.analyze_directory(str(self.project_root))

            # Run dependency analysis
            dependency_results = self.dependency_analyzer.analyze_directory(str(self.project_root))

            # Combine results
            self.interaction_data = {
                "timestamp": self.timestamp,
                "analysis_type": "interaction_mapping_for_dead_code",
                "results": {
                    "interactions": interaction_results.get("interactions", []),
                    "call_graph_analysis": call_graph_results,
                    "dependency_analysis": dependency_results
                }
            }

            # Initialize the interaction-aware analyzer with this data
            self.interaction_aware_analyzer = InteractionAwareDeadCodeAnalyzer(self.interaction_data)

            return self.interaction_data

        except Exception as e:
            print(f"⚠️  Warning: Interaction mapping analysis failed: {e}")
            return {}

    def _register_dead_code_plugins(self):
        """Register dead code analysis plugins using standardized batch registration."""
        try:
            # Import dead code analysis plugins
            from plugins.autoflake_fixer import AutoflakeFixer
            from plugins.creosote_analyzer import CreosoteAnalyzer
            from plugins.fawltydeps_analyzer import FawltyDepsAnalyzer

            # Use standardized batch registration
            plugin_classes = [AutoflakeFixer, CreosoteAnalyzer, FawltyDepsAnalyzer]
            self.register_plugins_batch(plugin_classes)

        except ImportError as e:
            self.logger.warning(f"Could not import dead code plugins: {e}")
    
    def run_basic_dead_code_analysis(self) -> Dict[str, Any]:
        """Run basic dead code analysis using Vulture."""
        print("\n" + "="*60)
        print("Running Basic Dead Code Analysis")
        print("="*60)
        
        try:
            results = self.dead_code_analyzer.analyze_dead_code(str(self.project_root))
            
            # Generate dead code report
            dead_code_report = {
                "timestamp": self.timestamp,
                "analysis_type": "basic_dead_code",
                "project_root": str(self.project_root),
                "total_issues": len(results.get("issues", [])),
                "dead_code_issues": len([i for i in results.get("issues", []) 
                                       if i.get("type") == "dead_code"]),
                "unreachable_code_issues": len([i for i in results.get("issues", []) 
                                              if i.get("type") == "unreachable_code"]),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"basic_dead_code_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(dead_code_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_issues": dead_code_report["total_issues"],
                "dead_code_count": dead_code_report["dead_code_issues"],
                "unreachable_code_count": dead_code_report["unreachable_code_issues"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_enhanced_dead_code_analysis(self) -> Dict[str, Any]:
        """Run enhanced dead code analysis with reduced false positives."""
        print("\n" + "="*60)
        print("Running Enhanced Dead Code Analysis")
        print("="*60)
        
        try:
            report = self.dead_code_analyzer.analyze_directory(str(self.project_root))
            
            # First, run interaction mapping analysis if enabled
            if self.use_interaction_mapping:
                interaction_data = self._run_interaction_mapping_analysis()
                if interaction_data:
                    print("✅ Interaction mapping data collected for enhanced analysis")
                else:
                    print("⚠️  Interaction mapping data not available, proceeding with standard analysis")

            # Convert report to dictionary format
            results = {
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_file": {str(k): v for k, v in report.issues_by_file.items()},
                "issues_by_severity": {str(k): v for k, v in report.issues_by_severity.items()},
                "issues_by_tool": {str(k): v for k, v in report.issues_by_tool.items()},
                "confidence_distribution": report.confidence_distribution,
                "potential_savings": report.potential_savings,
                "false_positives_filtered": report.false_positives_filtered,
                "impact_analysis": report.impact_analysis
            }

            # Enhance results with interaction mapping data if available
            if self.interaction_aware_analyzer:
                print("\n🔄 Enhancing dead code analysis with interaction mapping insights...")

                # Convert enhanced analyzer results to the format expected by interaction analyzer
                dead_code_results = {
                    "unused_functions": [],
                    "unused_classes": [],
                    "statistics": results
                }

                # Extract unused functions and classes from the enhanced analyzer results
                for issue_type, issues in results.get("issues_by_type", {}).items():
                    if "function" in issue_type.lower() or "unused" in issue_type.lower():
                        for issue in issues:
                            dead_code_results["unused_functions"].append({
                                "name": issue.get("name", ""),
                                "file": issue.get("file", ""),
                                "line": issue.get("line", 0)
                            })
                    elif "class" in issue_type.lower():
                        for issue in issues:
                            dead_code_results["unused_classes"].append({
                                "name": issue.get("name", ""),
                                "file": issue.get("file", ""),
                                "line": issue.get("line", 0)
                            })

                # Enhance with interaction mapping
                enhanced_results = self.interaction_aware_analyzer.enhance_dead_code_report(dead_code_results)

                # Update the main results with enhanced data
                results["unused_functions"] = enhanced_results.get("unused_functions", [])
                results["unused_classes"] = enhanced_results.get("unused_classes", [])
                results["false_positives_removed"] = enhanced_results.get("statistics", {}).get("false_positives_removed", 0)
                results["interaction_enhanced"] = True

                print(f"📊 Interaction-enhanced analysis completed:")
                print(f"   ✅ False positives removed: {results['false_positives_removed']}")
                print(f"   🔴 Confirmed unused functions: {len(results['unused_functions'])}")
                print(f"   🔴 Confirmed unused classes: {len(results['unused_classes'])}")
            
            # Generate enhanced dead code report
            enhanced_report = {
                "timestamp": self.timestamp,
                "analysis_type": "enhanced_dead_code",
                "project_root": str(self.project_root),
                "total_issues": report.total_issues,
                "high_confidence_issues": len([i for i in report.issues_by_severity.get("high", [])]),
                "cross_file_usage_checked": True,  # Enhanced analyzer always checks cross-file usage
                "interaction_mapping_enhanced": bool(self.interaction_aware_analyzer),
                "false_positives_removed": results.get("false_positives_removed", 0),
                "results": results
            }

            # Save report
            report_path = self.reports_dir / f"enhanced_dead_code_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(enhanced_report, f, indent=2)

            # Also save interaction mapping data if available
            if self.interaction_data:
                interaction_report_path = self.reports_dir / f"interaction_data_{self.timestamp}.json"
                with open(interaction_report_path, "w") as f:
                    json.dump(self.interaction_data, f, indent=2)

            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_issues": enhanced_report["total_issues"],
                "high_confidence_issues": enhanced_report["high_confidence_issues"],
                "interaction_enhanced": enhanced_report["interaction_mapping_enhanced"],
                "false_positives_removed": enhanced_report["false_positives_removed"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_unused_imports_analysis(self) -> Dict[str, Any]:
        """Run unused imports analysis."""
        print("\n" + "="*60)
        print("Running Unused Imports Analysis")
        print("="*60)
        
        try:
            results = self.import_analyzer.analyze_unused_imports(str(self.project_root))
            
            # Generate unused imports report
            unused_imports_report = {
                "timestamp": self.timestamp,
                "analysis_type": "unused_imports",
                "project_root": str(self.project_root),
                "total_unused_imports": len(results.get("unused_imports", [])),
                "files_with_unused_imports": len(set(i.get("file", "") for i in results.get("unused_imports", []))),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"unused_imports_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(unused_imports_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_unused_imports": unused_imports_report["total_unused_imports"],
                "files_affected": unused_imports_report["files_with_unused_imports"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_undefined_names_analysis(self) -> Dict[str, Any]:
        """Run undefined names analysis."""
        print("\n" + "="*60)
        print("Running Undefined Names Analysis")
        print("="*60)
        
        try:
            results = self.undefined_names_analyzer.analyze_undefined_names(str(self.project_root))
            
            # Generate undefined names report
            undefined_names_report = {
                "timestamp": self.timestamp,
                "analysis_type": "undefined_names",
                "project_root": str(self.project_root),
                "total_undefined_names": len(results.get("undefined_names", [])),
                "files_with_undefined_names": len(set(i.get("file", "") for i in results.get("undefined_names", []))),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"undefined_names_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(undefined_names_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_undefined_names": undefined_names_report["total_undefined_names"],
                "files_affected": undefined_names_report["files_with_undefined_names"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_auto_fix_dead_code(self) -> Dict[str, Any]:
        """Run automated dead code fixing."""
        print("\n" + "="*60)
        print("Running Automated Dead Code Fixing")
        print("="*60)
        
        # Auto-fixer is not available, return disabled status
        return {
            "status": "disabled",
            "message": "Auto-fixer functionality is not available in this version",
            "fixes_applied": 0,
            "files_modified": 0,
            "fixes_failed": 0
        }
    
    def run_plugin_analysis(self) -> Dict[str, Any]:
        """Run plugin-based dead code analysis."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}
        
        print("\n" + "="*60)
        print("Running Plugin-Based Dead Code Analysis")
        print("="*60)
        
        try:
            plugin_results = {}
            
            # Get dead code related plugins
            dead_code_plugins = [
                plugin for plugin in self.plugin_registry.list_plugins()
                if any(keyword in plugin.lower() for keyword in ["autoflake", "creosote", "fawltydeps"])
            ]
            
            for plugin_name in dead_code_plugins:
                try:
                    result = self.plugin_manager.execute_plugin(
                        plugin_name, 
                        {"project_root": str(self.project_root), "analysis_type": "dead_code"}
                    )
                    plugin_results[plugin_name] = result
                except Exception as e:
                    plugin_results[plugin_name] = {"status": "error", "error": str(e)}
            
            return {
                "status": "completed",
                "plugins_executed": len(plugin_results),
                "results": plugin_results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_dead_code_analysis(self) -> Dict[str, Any]:
        """Run comprehensive dead code analysis."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE DEAD CODE ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Plugins enabled: {self.enable_plugins}")
        
        total_start = time.time()
        
        # Run all dead code analyses
        self.results["basic_dead_code"] = self.run_basic_dead_code_analysis()
        self.results["enhanced_dead_code"] = self.run_enhanced_dead_code_analysis()
        self.results["unused_imports"] = self.run_unused_imports_analysis()
        self.results["undefined_names"] = self.run_undefined_names_analysis()
        self.results["auto_fix"] = self.run_auto_fix_dead_code()
        self.results["plugin_analysis"] = self.run_plugin_analysis()
        
        # Generate summary
        total_time = time.time() - total_start
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "analysis_categories": len(self.results) - 1,  # Exclude summary
            "plugins_enabled": self.enable_plugins,
            "status": "completed"
        }
        
        # Save comprehensive report
        report_path = self.reports_dir / f"dead_code_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("DEAD CODE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results


def main():
    """Main entry point for the dead code pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dead Code Analysis Pipeline - Comprehensive dead code detection and removal"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--disable-plugins",
        action="store_true",
        help="Disable plugin system"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["basic", "enhanced", "auto_fix", "all"],
        default="enhanced",
        help="Type of dead code analysis to perform (default: enhanced)"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Enable automatic fixing of dead code"
    )
    parser.add_argument(
        "--disable-interaction-mapping",
        action="store_true",
        help="Disable interaction mapping enhancement (uses only static analysis)"
    )
    
    args = parser.parse_args()
    
    pipeline = DeadCodePipeline(
        project_root=args.project_root,
        enable_plugins=not args.disable_plugins,
        use_interaction_mapping=not args.disable_interaction_mapping
    )
    
    if args.analysis_type == "all":
        results = pipeline.run_all_dead_code_analysis()
    elif args.analysis_type == "basic":
        results = pipeline.run_basic_dead_code_analysis()
    elif args.analysis_type == "enhanced":
        results = pipeline.run_enhanced_dead_code_analysis()
        # If auto-fix is requested, also run auto-fix
        if args.auto_fix:
            auto_fix_results = pipeline.run_auto_fix_dead_code()
            results["auto_fix"] = auto_fix_results
    elif args.analysis_type == "unused_imports":
        results = pipeline.run_unused_imports_analysis()
    elif args.analysis_type == "undefined_names":
        results = pipeline.run_undefined_names_analysis()
    elif args.analysis_type == "auto_fix":
        results = pipeline.run_auto_fix_dead_code()
    
    print(f"\nDead code pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()