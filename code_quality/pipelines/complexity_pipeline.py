#!/usr/bin/env python3
"""
Complexity Analysis Pipeline

Specialized pipeline for code complexity analysis with focus on cyclomatic complexity.
Supports multiple analysis types: cyclomatic, cognitive, maintainability, metrics.

Usage:
    python pipelines/complexity_pipeline.py --analysis-type cyclomatic
    python pipelines/complexity_pipeline.py --analysis-type cognitive
    python pipelines/complexity_pipeline.py --analysis-type maintainability
    python pipelines/complexity_pipeline.py --analysis-type metrics
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import complexity analyzers (ONLY complexity-related)
from analyzers.simple_complexity_analyzer import SimpleComplexityAnalyzer
from analyzers.metrics_analyzer import MetricsAnalyzer
from analyzers.architecture_analyzer import ArchitectureAnalyzer
from analyzers.call_graph_analyzer import CallGraphAnalyzer

# Import visualizers (ONLY complexity-related) - disabled for now due to dependency issues
VISUALIZERS_AVAILABLE = False
ComplexityHeatmapVisualizer = None
DashboardGenerator = None
print("Warning: Visualizers disabled due to missing dependencies")

# Import core components
from core.config import get_default_config
from plugins.plugin_registry import PluginRegistry
from plugins.plugin_manager import PluginManager


class ComplexityPipeline:
    """Specialized pipeline for complexity analysis."""
    
    def __init__(self, project_root: str = None, enable_plugins: bool = True):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.enable_plugins = enable_plugins
        
        # Initialize analyzers
        self.config = get_default_config()
        self.complexity_analyzer = SimpleComplexityAnalyzer(self.config)
        self.metrics_analyzer = MetricsAnalyzer(self.project_root)
        self.architecture_analyzer = ArchitectureAnalyzer(self.config)
        self.call_graph_analyzer = CallGraphAnalyzer(self.config)
        
        # Initialize visualizers
        if VISUALIZERS_AVAILABLE:
            self.complexity_heatmap = ComplexityHeatmapVisualizer()
            self.dashboard_generator = DashboardGenerator()
        else:
            self.complexity_heatmap = None
            self.dashboard_generator = None
        
        # Initialize plugin system
        if self.enable_plugins:
            try:
                self.plugin_registry = PluginRegistry()
                self.plugin_manager = PluginManager(self.plugin_registry)
                self._register_complexity_plugins()
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize plugin system: {e}")
                self.enable_plugins = False
        
        # Setup reports directory
        self.reports_dir = self.project_root / "code_quality" / "reports" / "complexity"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _register_complexity_plugins(self):
        """Register complexity-related plugins."""
        try:
            # Register complexity analysis plugins
            from plugins.creosote_analyzer import CreosoteAnalyzer
            from plugins.flake8_analyzer import Flake8Analyzer
            
            self.plugin_registry.register_plugin(CreosoteAnalyzer)
            self.plugin_registry.register_plugin(Flake8Analyzer)
            
            print(f"✅ Registered {len(self.plugin_registry.list_plugins())} complexity plugins")
        except ImportError as e:
            print(f"⚠️  Warning: Could not register some plugins: {e}")
    
    def run_cyclomatic_complexity_analysis(self) -> Dict[str, Any]:
        """Run cyclomatic complexity analysis."""
        print("\n" + "="*60)
        print("Running Cyclomatic Complexity Analysis")
        print("="*60)
        
        try:
            print(f"Analyzing directory: {self.project_root}")
            results = self.complexity_analyzer.analyze_directory(str(self.project_root))
            print(f"Analysis completed. Found {len(results)} files.")
            
            # Convert ModuleComplexity objects to dictionaries for JSON serialization
            serializable_results = {}
            for file_path, module_complexity in results.items():
                serializable_results[file_path] = {
                    "path": module_complexity.path,
                    "functions": [
                        {
                            "name": func.name,
                            "complexity": func.complexity,
                            "line_number": func.line_number,
                            "signature": func.signature
                        } for func in module_complexity.functions
                    ],
                    "classes": [
                        {
                            "name": cls.name,
                            "complexity": cls.complexity,
                            "line_number": cls.line_number,
                            "methods": [
                                {
                                    "name": method.name,
                                    "complexity": method.complexity,
                                    "line_number": method.line_number
                                } for method in cls.methods
                            ]
                        } for cls in module_complexity.classes
                    ],
                    "overall_metrics": module_complexity.overall_metrics,
                    "complexity_score": module_complexity.complexity_score
                }
            
            # Generate complexity report
            complexity_report = {
                "timestamp": self.timestamp,
                "analysis_type": "cyclomatic_complexity",
                "project_root": str(self.project_root),
                "total_files": len(results),
                "results": serializable_results
            }
            
            # Save report
            report_path = self.reports_dir / f"cyclomatic_complexity_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(complexity_report, f, indent=2)
            
            print(f"Report saved to: {report_path}")
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_files_analyzed": complexity_report["total_files"],
                "results": serializable_results
            }
        except Exception as e:
            print(f"Error in complexity analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def run_cognitive_complexity_analysis(self) -> Dict[str, Any]:
        """Run cognitive complexity analysis."""
        print("\n" + "="*60)
        print("Running Cognitive Complexity Analysis")
        print("="*60)
        
        try:
            # Use the available analyze_directory method
            results = self.complexity_analyzer.analyze_directory(str(self.project_root))
            
            # Extract cognitive complexity information from the results
            functions_data = []
            high_cognitive_complexity = 0
            
            for file_path, module_complexity in results.items():
                for func in module_complexity.functions:
                    # Use cyclomatic complexity as a proxy for cognitive complexity
                    cognitive_complexity = func.complexity
                    if cognitive_complexity > 15:
                        high_cognitive_complexity += 1
                    
                    functions_data.append({
                        "file": file_path,
                        "function": func.name,
                        "cognitive_complexity": cognitive_complexity,
                        "line_number": func.line_number
                    })
                
                # Also check class methods
                for cls in module_complexity.classes:
                    for method in cls.methods:
                        cognitive_complexity = method.complexity
                        if cognitive_complexity > 15:
                            high_cognitive_complexity += 1
                        
                        functions_data.append({
                            "file": file_path,
                            "function": f"{cls.name}.{method.name}",
                            "cognitive_complexity": cognitive_complexity,
                            "line_number": method.line_number
                        })
            
            # Generate cognitive complexity report
            cognitive_report = {
                "timestamp": self.timestamp,
                "analysis_type": "cognitive_complexity",
                "project_root": str(self.project_root),
                "total_functions": len(functions_data),
                "high_cognitive_complexity": high_cognitive_complexity,
                "results": {"functions": functions_data}
            }
            
            # Save report
            report_path = self.reports_dir / f"cognitive_complexity_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(cognitive_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "high_cognitive_complexity_count": high_cognitive_complexity,
                "results": cognitive_report["results"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_maintainability_analysis(self) -> Dict[str, Any]:
        """Run maintainability index analysis."""
        print("\n" + "="*60)
        print("Running Maintainability Index Analysis")
        print("="*60)
        
        try:
            # Use the complexity analyzer to get maintainability data
            results = self.complexity_analyzer.analyze_directory(str(self.project_root))
            
            # Extract maintainability information
            files_data = []
            total_maintainability = 0
            low_maintainability_count = 0
            
            for file_path, module_complexity in results.items():
                maintainability_index = module_complexity.overall_metrics["maintainability_index"]
                total_maintainability += maintainability_index
                
                if maintainability_index < 20:
                    low_maintainability_count += 1
                
                files_data.append({
                    "file": file_path,
                    "maintainability_index": maintainability_index,
                    "cyclomatic_complexity": module_complexity.overall_metrics["cyclomatic_complexity"],
                    "halstead_volume": module_complexity.overall_metrics["halstead_volume"]
                })
            
            average_maintainability = total_maintainability / len(files_data) if files_data else 0
            
            # Generate maintainability report
            maintainability_report = {
                "timestamp": self.timestamp,
                "analysis_type": "maintainability_index",
                "project_root": str(self.project_root),
                "average_maintainability": average_maintainability,
                "low_maintainability_files": low_maintainability_count,
                "results": {"files": files_data}
            }
            
            # Save report
            report_path = self.reports_dir / f"maintainability_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(maintainability_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "average_maintainability": average_maintainability,
                "low_maintainability_count": low_maintainability_count,
                "results": maintainability_report["results"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_architecture_complexity_analysis(self) -> Dict[str, Any]:
        """Run architecture complexity analysis."""
        print("\n" + "="*60)
        print("Running Architecture Complexity Analysis")
        print("="*60)
        
        try:
            # Use the available analyze_directory method
            results = self.architecture_analyzer.analyze_directory(str(self.project_root))
            
            # Generate architecture report
            architecture_report = {
                "timestamp": self.timestamp,
                "analysis_type": "architecture_complexity",
                "project_root": str(self.project_root),
                "total_modules": len(results),
                "circular_dependencies": 0,  # Simplified - would need more complex analysis
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"architecture_complexity_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(architecture_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "circular_dependencies_count": 0,
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_call_graph_analysis(self) -> Dict[str, Any]:
        """Run call graph complexity analysis."""
        print("\n" + "="*60)
        print("Running Call Graph Analysis")
        print("="*60)
        
        try:
            # Use the available analyze_directory method
            results = self.call_graph_analyzer.analyze_directory(str(self.project_root))
            
            # Generate call graph report
            call_graph_report = {
                "timestamp": self.timestamp,
                "analysis_type": "call_graph_complexity",
                "project_root": str(self.project_root),
                "total_functions": len(results),
                "max_call_depth": 0,  # Simplified - would need more complex analysis
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"call_graph_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(call_graph_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "max_call_depth": 0,
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_plugin_analysis(self) -> Dict[str, Any]:
        """Run plugin-based complexity analysis."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}
        
        print("\n" + "="*60)
        print("Running Plugin-Based Complexity Analysis")
        print("="*60)
        
        try:
            plugin_results = {}
            
            # Get complexity-related plugins
            try:
                from plugins.base_plugin import PluginCategory
                complexity_plugins = self.plugin_registry.get_plugins_by_category(PluginCategory.ANALYSIS)
            except ImportError:
                # Fallback to getting all plugins
                complexity_plugins = list(self.plugin_registry.list_plugins().keys())
            
            for plugin_name in complexity_plugins:
                try:
                    result = self.plugin_manager.execute_plugin(
                        plugin_name, 
                        {"project_root": str(self.project_root), "analysis_type": "complexity"}
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
    
    def run_complexity_visualization(self) -> Dict[str, Any]:
        """Run complexity visualization."""
        print("\n" + "="*60)
        print("Running Complexity Visualization")
        print("="*60)
        
        if not VISUALIZERS_AVAILABLE:
            return {"status": "disabled", "message": "Visualizers not available - missing dependencies"}
        
        try:
            # Generate complexity heatmap
            heatmap_path = self.reports_dir / f"complexity_heatmap_{self.timestamp}.html"
            self.complexity_heatmap.generate_heatmap(str(self.project_root), str(heatmap_path))
            
            # Generate complexity dashboard
            dashboard_path = self.reports_dir / f"complexity_dashboard_{self.timestamp}.html"
            self.dashboard_generator.generate_complexity_dashboard(
                str(self.project_root), 
                str(dashboard_path)
            )
            
            return {
                "status": "completed",
                "heatmap_path": str(heatmap_path),
                "dashboard_path": str(dashboard_path)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_complexity_analysis(self) -> Dict[str, Any]:
        """Run comprehensive complexity analysis."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE COMPLEXITY ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Plugins enabled: {self.enable_plugins}")
        
        total_start = time.time()
        
        # Run all complexity analyses
        self.results["cyclomatic_complexity"] = self.run_cyclomatic_complexity_analysis()
        self.results["cognitive_complexity"] = self.run_cognitive_complexity_analysis()
        self.results["maintainability_analysis"] = self.run_maintainability_analysis()
        self.results["architecture_complexity"] = self.run_architecture_complexity_analysis()
        self.results["call_graph_analysis"] = self.run_call_graph_analysis()
        self.results["plugin_analysis"] = self.run_plugin_analysis()
        self.results["visualization"] = self.run_complexity_visualization()
        
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
        report_path = self.reports_dir / f"complexity_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("COMPLEXITY ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results


def main():
    """Main entry point for the complexity pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complexity Analysis Pipeline - Comprehensive code complexity assessment"
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
        choices=["cyclomatic", "cognitive", "maintainability", "architecture", "call_graph", "all"],
        default="all",
        help="Specific complexity analysis to run (default: all)"
    )
    
    args = parser.parse_args()
    
    pipeline = ComplexityPipeline(
        project_root=args.project_root,
        enable_plugins=not args.disable_plugins
    )
    
    if args.analysis_type == "all":
        results = pipeline.run_all_complexity_analysis()
    elif args.analysis_type == "cyclomatic":
        results = pipeline.run_cyclomatic_complexity_analysis()
    elif args.analysis_type == "cognitive":
        results = pipeline.run_cognitive_complexity_analysis()
    elif args.analysis_type == "maintainability":
        results = pipeline.run_maintainability_analysis()
    elif args.analysis_type == "architecture":
        results = pipeline.run_architecture_complexity_analysis()
    elif args.analysis_type == "call_graph":
        results = pipeline.run_call_graph_analysis()
    
    print(f"\nComplexity pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()