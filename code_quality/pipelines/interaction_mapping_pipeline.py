#!/usr/bin/env python3
"""
Code Interaction Mapping Pipeline

Specialized pipeline for code interaction analysis including:
- Function call mapping
- Class interaction mapping
- Module dependency mapping
- Data flow analysis
- Interaction visualization
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import interaction mappers (ONLY interaction mapping related)
from mappers.map_code_interactions import CodeInteractionMapper
from mappers.enhanced_map_code_interactions import EnhancedCodeInteractionMapper
# from mappers.visualize_interactions import InteractionVisualizer  # Class doesn't exist

# Import analyzers for interaction analysis (ONLY interaction mapping related)
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.enhanced_dependency_analyzer import EnhancedDependencyAnalyzer
from analyzers.data_flow_analyzer import DataFlowAnalyzer
from analyzers.call_graph_analyzer import CallGraphAnalyzer
from analyzers.architecture_analyzer import ArchitectureAnalyzer

# Import visualizers (ONLY interaction mapping related)
from visualizers.interaction_network import InteractionNetworkVisualizer
from visualizers.dependency_graph import DependencyGraphVisualizer
from visualizers.dashboard_generator import DashboardGenerator

# Import scripts (ONLY interaction mapping related)
from scripts.extract_interactions import ExtractInteractions
from scripts.interaction_summary import InteractionSummary

# Note: CodeInteractionMapperPipeline was removed as it was redundant with mappers

# Import core components
from core.config import get_default_config
from plugins.plugin_registry import PluginRegistry
from plugins.plugin_manager import PluginManager


class InteractionMappingPipeline:
    """Specialized pipeline for code interaction mapping."""
    
    def __init__(self, project_root: str = None, enable_plugins: bool = True):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.enable_plugins = enable_plugins
        
        # Initialize interaction mappers
        self.code_interaction_mapper = CodeInteractionMapper(str(self.project_root))
        self.enhanced_interaction_mapper = EnhancedCodeInteractionMapper(str(self.project_root))
        
        # Initialize analyzers
        self.config = get_default_config()
        self.call_graph_analyzer = CallGraphAnalyzer(self.config)
        self.dependency_analyzer = DependencyAnalyzer(self.config)
        self.data_flow_analyzer = DataFlowAnalyzer(str(self.project_root))
        self.architecture_analyzer = ArchitectureAnalyzer(self.config)
        
        # Initialize visualizers
        self.interaction_network_visualizer = InteractionNetworkVisualizer()
        self.dependency_graph_visualizer = DependencyGraphVisualizer()
        self.dashboard_generator = DashboardGenerator()
        
        # Initialize plugin system
        if self.enable_plugins:
            self.plugin_registry = PluginRegistry()
            self.plugin_manager = PluginManager(self.plugin_registry)
            self._register_interaction_plugins()
        
        # Setup reports directory
        self.reports_dir = self.project_root / "code_quality" / "reports" / "interaction_mapping"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _register_interaction_plugins(self):
        """Register interaction mapping plugins."""
        try:
            # Register analysis plugins that can help with interaction mapping
            from plugins.creosote_analyzer import CreosoteAnalyzer
            from plugins.fawltydeps_analyzer import FawltyDepsAnalyzer
            
            self.plugin_registry.register_plugin(CreosoteAnalyzer)
            self.plugin_registry.register_plugin(FawltyDepsAnalyzer)
            
            print(f"✅ Registered {len(self.plugin_registry.list_plugins())} interaction mapping plugins")
        except ImportError as e:
            print(f"⚠️  Warning: Could not register some plugins: {e}")
    
    def run_basic_interaction_mapping(self) -> Dict[str, Any]:
        """Run basic code interaction mapping."""
        print("\n" + "="*60)
        print("Running Basic Code Interaction Mapping")
        print("="*60)
        
        try:
            results = self.code_interaction_mapper.map_interactions(str(self.project_root))
            
            # Generate interaction mapping report
            interaction_report = {
                "timestamp": self.timestamp,
                "analysis_type": "basic_interaction_mapping",
                "project_root": str(self.project_root),
                "total_interactions": len(results.get("interactions", [])),
                "function_calls": len([i for i in results.get("interactions", []) 
                                     if i.get("type") == "function_call"]),
                "class_interactions": len([i for i in results.get("interactions", []) 
                                         if i.get("type") == "class_interaction"]),
                "module_dependencies": len(results.get("module_dependencies", [])),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"basic_interaction_mapping_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(interaction_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_interactions": interaction_report["total_interactions"],
                "function_calls": interaction_report["function_calls"],
                "class_interactions": interaction_report["class_interactions"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_enhanced_interaction_mapping(self) -> Dict[str, Any]:
        """Run enhanced code interaction mapping."""
        print("\n" + "="*60)
        print("Running Enhanced Code Interaction Mapping")
        print("="*60)
        
        try:
            results = self.enhanced_interaction_mapper.map_interactions(str(self.project_root))
            
            # Generate enhanced interaction mapping report
            enhanced_report = {
                "timestamp": self.timestamp,
                "analysis_type": "enhanced_interaction_mapping",
                "project_root": str(self.project_root),
                "total_interactions": len(results.get("interactions", [])),
                "complex_interactions": len([i for i in results.get("interactions", []) 
                                           if i.get("complexity", 0) > 5]),
                "cross_module_interactions": len([i for i in results.get("interactions", []) 
                                                if i.get("cross_module", False)]),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"enhanced_interaction_mapping_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(enhanced_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_interactions": enhanced_report["total_interactions"],
                "complex_interactions": enhanced_report["complex_interactions"],
                "cross_module_interactions": enhanced_report["cross_module_interactions"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_call_graph_analysis(self) -> Dict[str, Any]:
        """Run call graph analysis."""
        print("\n" + "="*60)
        print("Running Call Graph Analysis")
        print("="*60)
        
        try:
            results = self.call_graph_analyzer.analyze_call_graph(str(self.project_root))
            
            # Generate call graph report
            call_graph_report = {
                "timestamp": self.timestamp,
                "analysis_type": "call_graph_analysis",
                "project_root": str(self.project_root),
                "total_functions": results.get("total_functions", 0),
                "max_call_depth": results.get("max_call_depth", 0),
                "circular_calls": len(results.get("circular_calls", [])),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"call_graph_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(call_graph_report, f, indent=2)
            
            # Generate interaction analysis data for the summary script
            self._generate_interaction_analysis_data(results)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_functions": call_graph_report["total_functions"],
                "max_call_depth": call_graph_report["max_call_depth"],
                "circular_calls": call_graph_report["circular_calls"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _generate_interaction_analysis_data(self, results: Dict[str, Any]) -> None:
        """Generate interaction analysis data file for the summary script."""
        try:
            # Create interaction analysis data structure
            interaction_data = {
                "summary": {
                    "files_processed": results.get("total_functions", 0),
                    "total_issues": len(results.get("issues", [])),
                    "undefined_functions": len([i for i in results.get("issues", []) if i.get("type") == "undefined_function"]),
                    "missing_await": len([i for i in results.get("issues", []) if i.get("type") == "missing_await"])
                },
                "issues": results.get("issues", [])
            }
            
            # Save to the expected location
            interaction_file = Path("/workspace/code_quality/interaction_analysis.json")
            with open(interaction_file, "w") as f:
                json.dump(interaction_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not generate interaction analysis data: {e}")
    
    def run_dependency_analysis(self) -> Dict[str, Any]:
        """Run dependency analysis."""
        print("\n" + "="*60)
        print("Running Dependency Analysis")
        print("="*60)
        
        try:
            results = self.dependency_analyzer.analyze_dependencies(str(self.project_root))
            
            # Generate dependency report
            dependency_report = {
                "timestamp": self.timestamp,
                "analysis_type": "dependency_analysis",
                "project_root": str(self.project_root),
                "total_dependencies": len(results.get("dependencies", [])),
                "external_dependencies": len([d for d in results.get("dependencies", []) 
                                            if d.get("type") == "external"]),
                "internal_dependencies": len([d for d in results.get("dependencies", []) 
                                            if d.get("type") == "internal"]),
                "circular_dependencies": len(results.get("circular_dependencies", [])),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"dependency_analysis_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(dependency_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_dependencies": dependency_report["total_dependencies"],
                "external_dependencies": dependency_report["external_dependencies"],
                "internal_dependencies": dependency_report["internal_dependencies"],
                "circular_dependencies": dependency_report["circular_dependencies"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_data_flow_analysis(self) -> Dict[str, Any]:
        """Run data flow analysis."""
        print("\n" + "="*60)
        print("Running Data Flow Analysis")
        print("="*60)
        
        try:
            results = self.data_flow_analyzer.analyze_data_flow(str(self.project_root))
            
            # Generate data flow report
            data_flow_report = {
                "timestamp": self.timestamp,
                "analysis_type": "data_flow_analysis",
                "project_root": str(self.project_root),
                "data_flows": len(results.get("data_flows", [])),
                "complex_data_flows": len([df for df in results.get("data_flows", []) 
                                         if df.get("complexity", 0) > 3]),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"data_flow_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(data_flow_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "data_flows": data_flow_report["data_flows"],
                "complex_data_flows": data_flow_report["complex_data_flows"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_architecture_analysis(self) -> Dict[str, Any]:
        """Run architecture analysis."""
        print("\n" + "="*60)
        print("Running Architecture Analysis")
        print("="*60)
        
        try:
            results = self.architecture_analyzer.analyze_architecture(str(self.project_root))
            
            # Generate architecture report
            architecture_report = {
                "timestamp": self.timestamp,
                "analysis_type": "architecture_analysis",
                "project_root": str(self.project_root),
                "modules": len(results.get("modules", [])),
                "layers": len(results.get("layers", [])),
                "violations": len(results.get("violations", [])),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"architecture_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(architecture_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "modules": architecture_report["modules"],
                "layers": architecture_report["layers"],
                "violations": architecture_report["violations"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_interaction_visualization(self) -> Dict[str, Any]:
        """Run interaction visualization."""
        print("\n" + "="*60)
        print("Running Interaction Visualization")
        print("="*60)
        
        try:
            # Generate interaction visualizations
            visualization_results = {}
            
            # Basic interaction visualization
            basic_viz_path = self.reports_dir / f"interaction_visualization_{self.timestamp}.html"
            self.interaction_visualizer.visualize_interactions(str(self.project_root), str(basic_viz_path))
            visualization_results["basic_visualization"] = str(basic_viz_path)
            
            # Interaction network visualization
            network_viz_path = self.reports_dir / f"interaction_network_{self.timestamp}.html"
            self.interaction_network.generate_network(str(self.project_root), str(network_viz_path))
            visualization_results["network_visualization"] = str(network_viz_path)
            
            # Dependency graph visualization
            dep_graph_path = self.reports_dir / f"dependency_graph_{self.timestamp}.html"
            self.dependency_graph.generate_graph(str(self.project_root), str(dep_graph_path))
            visualization_results["dependency_graph"] = str(dep_graph_path)
            
            # Dashboard generation
            dashboard_path = self.reports_dir / f"interaction_dashboard_{self.timestamp}.html"
            self.dashboard_generator.generate_interaction_dashboard(
                str(self.project_root), 
                str(dashboard_path)
            )
            visualization_results["dashboard"] = str(dashboard_path)
            
            return {
                "status": "completed",
                "visualizations": visualization_results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_plugin_analysis(self) -> Dict[str, Any]:
        """Run plugin-based interaction analysis."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}
        
        print("\n" + "="*60)
        print("Running Plugin-Based Interaction Analysis")
        print("="*60)
        
        try:
            plugin_results = {}
            
            # Get analysis plugins
            analysis_plugins = self.plugin_registry.get_plugins_by_category(PluginCategory.ANALYSIS)
            
            for plugin_name in analysis_plugins:
                try:
                    result = self.plugin_manager.execute_plugin(
                        plugin_name, 
                        {"project_root": str(self.project_root), "analysis_type": "interaction_mapping"}
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
    
    def run_all_interaction_mapping(self) -> Dict[str, Any]:
        """Run comprehensive interaction mapping."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE INTERACTION MAPPING PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Plugins enabled: {self.enable_plugins}")
        
        total_start = time.time()
        
        # Run all interaction mapping analyses
        self.results["basic_interaction_mapping"] = self.run_basic_interaction_mapping()
        self.results["enhanced_interaction_mapping"] = self.run_enhanced_interaction_mapping()
        self.results["call_graph_analysis"] = self.run_call_graph_analysis()
        self.results["dependency_analysis"] = self.run_dependency_analysis()
        self.results["data_flow_analysis"] = self.run_data_flow_analysis()
        self.results["architecture_analysis"] = self.run_architecture_analysis()
        self.results["visualization"] = self.run_interaction_visualization()
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
        report_path = self.reports_dir / f"interaction_mapping_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("INTERACTION MAPPING COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results


def main():
    """Main entry point for the interaction mapping pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interaction Mapping Pipeline - Comprehensive code interaction analysis"
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
        choices=["call_graph", "dependencies", "data_flow", "architecture", "visualization"],
        default="call_graph",
        help="Type of interaction analysis to perform (default: call_graph)"
    )
    
    args = parser.parse_args()
    
    pipeline = InteractionMappingPipeline(
        project_root=args.project_root,
        enable_plugins=not args.disable_plugins
    )
    
    if args.analysis_type == "all":
        results = pipeline.run_all_interaction_mapping()
    elif args.analysis_type == "basic":
        results = pipeline.run_basic_interaction_mapping()
    elif args.analysis_type == "enhanced":
        results = pipeline.run_enhanced_interaction_mapping()
    elif args.analysis_type == "call_graph":
        results = pipeline.run_call_graph_analysis()
    elif args.analysis_type == "dependencies":
        results = pipeline.run_dependency_analysis()
    elif args.analysis_type == "data_flow":
        results = pipeline.run_data_flow_analysis()
    elif args.analysis_type == "architecture":
        results = pipeline.run_architecture_analysis()
    elif args.analysis_type == "visualization":
        results = pipeline.run_interaction_visualization()
    
    print(f"\nInteraction mapping pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()