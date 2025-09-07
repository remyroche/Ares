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
import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set
from collections import defaultdict

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
from plugins.base_plugin import PluginCategory


class GraphBasedDeadCodeAnalyzer:
    """Graph-based dead code analyzer integrated with interaction mapping."""
    
    def __init__(self, interaction_data: Dict[str, Any]):
        self.interaction_data = interaction_data
        self.used_functions = set()
        self.used_classes = set()
        self.used_methods = set()
        self.import_graph = defaultdict(set)
        self.call_graph = defaultdict(set)
        
    def build_usage_graphs(self):
        """Build usage graphs from interaction data."""
        print("🔗 Building usage graphs from interaction data...")
        
        interactions = self.interaction_data.get('results', {}).get('interactions', [])
        
        for interaction in interactions:
            interaction_type = interaction.get('type', '')
            source = interaction.get('source', '')
            target = interaction.get('target', '')
            source_file = interaction.get('source_file', '')
            
            if interaction_type == 'function_call':
                # Add to call graph
                self.call_graph[source].add(target)
                self.used_functions.add(target)
                
                # If it's a method call, track the class
                if '.' in target:
                    class_name = target.split('.')[0]
                    self.used_classes.add(class_name)
                    self.used_methods.add(target)
            
            elif interaction_type == 'class_instantiation':
                self.used_classes.add(target)
                self.call_graph[source].add(target)
            
            elif interaction_type == 'import':
                # Build import graph
                if source_file and target:
                    self.import_graph[source_file].add(target)
                    if '.' in target:
                        class_name = target.split('.')[-1]
                        self.used_classes.add(class_name)
                    else:
                        self.used_functions.add(target)
        
        print(f"✅ Built usage graphs:")
        print(f"   📊 {len(self.used_functions)} used functions")
        print(f"   📊 {len(self.used_classes)} used classes")
        print(f"   📊 {len(self.used_methods)} used methods")
        print(f"   📊 {len(self.call_graph)} call graph nodes")
        print(f"   📊 {len(self.import_graph)} import graph nodes")
    
    def analyze_dead_code(self, project_root: Path) -> Dict[str, Any]:
        """Analyze dead code using graph-based approach."""
        print("🔍 Analyzing dead code using graph-based approach...")
        
        python_files = list(project_root.rglob("*.py"))
        dead_code_results = {
            "unused_functions": [],
            "unused_classes": [],
            "unused_methods": [],
            "orphaned_files": [],
            "dead_imports": [],
            "statistics": {
                "total_files": len(python_files),
                "files_with_dead_code": 0,
                "total_unused_functions": 0,
                "total_unused_classes": 0,
                "total_unused_methods": 0
            }
        }
        
        for file_path in python_files:
            if "test" in str(file_path) or "__pycache__" in str(file_path):
                continue
            
            file_dead_code = self._analyze_file_dead_code(file_path)
            
            if file_dead_code:
                dead_code_results["statistics"]["files_with_dead_code"] += 1
                dead_code_results["statistics"]["total_unused_functions"] += len(file_dead_code.get("unused_functions", []))
                dead_code_results["statistics"]["total_unused_classes"] += len(file_dead_code.get("unused_classes", []))
                dead_code_results["statistics"]["total_unused_methods"] += len(file_dead_code.get("unused_methods", []))
                
                # Add to results
                dead_code_results["unused_functions"].extend(file_dead_code.get("unused_functions", []))
                dead_code_results["unused_classes"].extend(file_dead_code.get("unused_classes", []))
                dead_code_results["unused_methods"].extend(file_dead_code.get("unused_methods", []))
        
        return dead_code_results
    
    def _analyze_file_dead_code(self, file_path: Path) -> Dict[str, List]:
        """Analyze dead code in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract defined entities
            defined_functions = set()
            defined_classes = set()
            defined_methods = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_functions.add(node.name)
                    # Check if it's a method
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef) and hasattr(parent, 'body') and node in parent.body:
                            method_name = f"{parent.name}.{node.name}"
                            defined_methods.add(method_name)
                            break
                elif isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)
            
            # Find unused entities
            unused_functions = []
            unused_classes = []
            unused_methods = []
            
            for func in defined_functions:
                if func not in self.used_functions and not self._is_special_function(func):
                    unused_functions.append({
                        "name": func,
                        "file": str(file_path),
                        "line": self._get_function_line(tree, func)
                    })
            
            for cls in defined_classes:
                if cls not in self.used_classes and not self._is_special_class(cls):
                    unused_classes.append({
                        "name": cls,
                        "file": str(file_path),
                        "line": self._get_class_line(tree, cls)
                    })
            
            for method in defined_methods:
                if method not in self.used_methods and not self._is_special_method(method):
                    unused_methods.append({
                        "name": method,
                        "file": str(file_path),
                        "line": self._get_method_line(tree, method)
                    })
            
            result = {}
            if unused_functions:
                result["unused_functions"] = unused_functions
            if unused_classes:
                result["unused_classes"] = unused_classes
            if unused_methods:
                result["unused_methods"] = unused_methods
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return {}
    
    def _is_special_function(self, func_name: str) -> bool:
        """Check if a function is special (like __init__, main, etc.)."""
        special_functions = {
            '__init__', '__main__', 'main', 'setup', 'teardown', 
            'test_', 'run_', 'execute', 'if __name__ == "__main__"'
        }
        return any(func_name.startswith(special) for special in special_functions)
    
    def _is_special_class(self, class_name: str) -> bool:
        """Check if a class is special (like base classes, etc.)."""
        special_classes = {
            'Base', 'Abstract', 'Interface', 'Protocol', 'Exception', 'Error'
        }
        return any(class_name.startswith(special) for special in special_classes)
    
    def _is_special_method(self, method_name: str) -> bool:
        """Check if a method is special (like __init__, __str__, etc.)."""
        special_methods = {
            '__init__', '__str__', '__repr__', '__len__', '__getitem__',
            '__setitem__', '__delitem__', '__iter__', '__next__', '__enter__',
            '__exit__', '__call__', '__eq__', '__ne__', '__lt__', '__le__',
            '__gt__', '__ge__', '__hash__', '__bool__'
        }
        return any(method_name.endswith(special) for special in special_methods)
    
    def _get_function_line(self, tree: ast.AST, func_name: str) -> int:
        """Get line number of a function definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node.lineno
        return 0
    
    def _get_class_line(self, tree: ast.AST, class_name: str) -> int:
        """Get line number of a class definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node.lineno
        return 0
    
    def _get_method_line(self, tree: ast.AST, method_name: str) -> int:
        """Get line number of a method definition."""
        class_name, method_name = method_name.split('.', 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == method_name:
                        return child.lineno
        return 0


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
        
        # Initialize dead code analyzer (will be set after interaction mapping)
        self.dead_code_analyzer = None
    
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
            results = self.call_graph_analyzer.analyze_directory(str(self.project_root))
            
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
            interaction_file = self.project_root / "code_quality" / "interaction_analysis.json"
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
            results = self.dependency_analyzer.analyze_directory(str(self.project_root))
            
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
            results = self.architecture_analyzer.analyze_directory(str(self.project_root))
            
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
            self._generate_basic_visualization(str(basic_viz_path))
            visualization_results["basic_visualization"] = str(basic_viz_path)
            
            # Interaction network visualization
            network_viz_path = self.reports_dir / f"interaction_network_{self.timestamp}.html"
            self._generate_network_visualization(str(network_viz_path))
            visualization_results["network_visualization"] = str(network_viz_path)
            
            # Dependency graph visualization
            dep_graph_path = self.reports_dir / f"dependency_graph_{self.timestamp}.html"
            self._generate_dependency_visualization(str(dep_graph_path))
            visualization_results["dependency_graph"] = str(dep_graph_path)
            
            # Dashboard generation
            dashboard_path = self.reports_dir / f"interaction_dashboard_{self.timestamp}.html"
            self._generate_dashboard(str(dashboard_path))
            visualization_results["dashboard"] = str(dashboard_path)
            
            return {
                "status": "completed",
                "visualizations": visualization_results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _generate_basic_visualization(self, output_path: str) -> None:
        """Generate basic interaction visualization."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Interaction Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Interaction Visualization</h1>
        <p>Generated on: {self.timestamp}</p>
        <p>Project: {self.project_root}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Summary</h2>
        <div class="metric">Total Functions: {self.results.get('call_graph_analysis', {}).get('total_functions', 0)}</div>
        <div class="metric">Max Call Depth: {self.results.get('call_graph_analysis', {}).get('max_call_depth', 0)}</div>
        <div class="metric">Circular Calls: {self.results.get('call_graph_analysis', {}).get('circular_calls', 0)}</div>
    </div>
    
    <div class="section">
        <h2>Dependencies</h2>
        <div class="metric">Total Dependencies: {self.results.get('dependency_analysis', {}).get('total_dependencies', 0)}</div>
        <div class="metric">External Dependencies: {self.results.get('dependency_analysis', {}).get('external_dependencies', 0)}</div>
        <div class="metric">Internal Dependencies: {self.results.get('dependency_analysis', {}).get('internal_dependencies', 0)}</div>
    </div>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_network_visualization(self, output_path: str) -> None:
        """Generate network visualization."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interaction Network Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .node {{ stroke: #fff; stroke-width: 1.5px; }}
        .link {{ stroke: #999; stroke-opacity: .6; }}
    </style>
</head>
<body>
    <h1>Interaction Network</h1>
    <div id="network"></div>
    <script>
        // Simple network visualization placeholder
        const svg = d3.select("#network").append("svg")
            .attr("width", 800)
            .attr("height", 600);
        
        svg.append("text")
            .attr("x", 400)
            .attr("y", 300)
            .attr("text-anchor", "middle")
            .text("Network visualization would be rendered here");
    </script>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_dependency_visualization(self, output_path: str) -> None:
        """Generate dependency graph visualization."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dependency Graph Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dependency {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Dependency Graph</h1>
    <div id="dependencies">
        <p>Dependency visualization would be rendered here based on analysis results.</p>
        <p>Total modules analyzed: {self.results.get('dependency_analysis', {}).get('total_modules', 0)}</p>
    </div>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_dashboard(self, output_path: str) -> None:
        """Generate comprehensive dashboard."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interaction Mapping Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; border: 1px solid #ddd; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .label {{ color: #666; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>Interaction Mapping Dashboard</h1>
    <p>Generated on: {self.timestamp}</p>
    
    <div class="dashboard">
        <div class="card">
            <div class="label">Total Functions</div>
            <div class="metric">{self.results.get('call_graph_analysis', {}).get('total_functions', 0)}</div>
        </div>
        
        <div class="card">
            <div class="label">Max Call Depth</div>
            <div class="metric">{self.results.get('call_graph_analysis', {}).get('max_call_depth', 0)}</div>
        </div>
        
        <div class="card">
            <div class="label">Total Dependencies</div>
            <div class="metric">{self.results.get('dependency_analysis', {}).get('total_dependencies', 0)}</div>
        </div>
        
        <div class="card">
            <div class="label">Circular Dependencies</div>
            <div class="metric">{len(self.results.get('dependency_analysis', {}).get('circular_dependencies', []))}</div>
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
    
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
        # Run dead code analysis if we have interaction data
        if self.results.get('basic_interaction_mapping'):
            self._run_dead_code_analysis()
        
        print("INTERACTION MAPPING COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results
    
    def _run_dead_code_analysis(self):
        """Run the integrated dead code analysis."""
        print("\n" + "="*60)
        print("Running Graph-Based Dead Code Analysis")
        print("="*60)
        
        # Initialize dead code analyzer with interaction data
        self.dead_code_analyzer = GraphBasedDeadCodeAnalyzer(
            self.results['basic_interaction_mapping']
        )
        self.dead_code_analyzer.build_usage_graphs()
        
        # Run dead code analysis
        dead_code_results = self.dead_code_analyzer.analyze_dead_code(self.project_root)
        self.results['dead_code_analysis'] = dead_code_results
        
        # Print summary
        stats = dead_code_results['statistics']
        print(f"\n📊 DEAD CODE ANALYSIS SUMMARY:")
        print(f"  🔴 Total unused functions: {stats['total_unused_functions']}")
        print(f"  🔴 Total unused classes: {stats['total_unused_classes']}")
        print(f"  🔴 Total unused methods: {stats['total_unused_methods']}")
        print(f"  📁 Files with dead code: {stats['files_with_dead_code']}")
        print(f"  📁 Total files analyzed: {stats['total_files']}")
        
        if stats['total_files'] > 0:
            dead_code_percentage = (stats['total_unused_functions'] + stats['total_unused_classes']) / stats['total_files'] * 100
            print(f"  📈 Dead code percentage: {dead_code_percentage:.1f}%")
        
        # Generate dead code cleanup recommendations
        self._generate_dead_code_cleanup_report(dead_code_results)
    
    def _generate_dead_code_cleanup_report(self, dead_code_data: Dict[str, Any]):
        """Generate dead code cleanup recommendations."""
        # Group dead code by file
        files_with_dead_code = defaultdict(list)
        
        for func in dead_code_data.get('unused_functions', []):
            files_with_dead_code[func['file']].append(f"Function: {func['name']} (line {func['line']})")
        
        for cls in dead_code_data.get('unused_classes', []):
            files_with_dead_code[cls['file']].append(f"Class: {cls['name']} (line {cls['line']})")
        
        for method in dead_code_data.get('unused_methods', []):
            files_with_dead_code[method['file']].append(f"Method: {method['name']} (line {method['line']})")
        
        # Generate cleanup report
        cleanup_report = {
            "timestamp": self.timestamp,
            "summary": dead_code_data['statistics'],
            "files_with_dead_code": dict(files_with_dead_code),
            "recommendations": {
                "high_priority": [],
                "medium_priority": [],
                "low_priority": []
            }
        }
        
        # Categorize recommendations
        for file_path, dead_items in files_with_dead_code.items():
            if len(dead_items) > 20:
                cleanup_report["recommendations"]["high_priority"].append({
                    "file": file_path,
                    "dead_items_count": len(dead_items),
                    "items": dead_items[:10]  # Show first 10
                })
            elif len(dead_items) > 10:
                cleanup_report["recommendations"]["medium_priority"].append({
                    "file": file_path,
                    "dead_items_count": len(dead_items),
                    "items": dead_items[:5]  # Show first 5
                })
            else:
                cleanup_report["recommendations"]["low_priority"].append({
                    "file": file_path,
                    "dead_items_count": len(dead_items),
                    "items": dead_items
                })
        
        # Save cleanup report
        cleanup_file = self.reports_dir / f"dead_code_cleanup_recommendations_{self.timestamp}.json"
        with open(cleanup_file, 'w') as f:
            json.dump(cleanup_report, f, indent=2, default=str)
        
        print(f"✅ Generated dead code cleanup report: {cleanup_file}")


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
        choices=["call_graph", "dependencies", "data_flow", "architecture", "visualization", "all"],
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