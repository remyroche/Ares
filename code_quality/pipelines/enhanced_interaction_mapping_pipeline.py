#!/usr/bin/env python3
"""
Enhanced Code Interaction Mapping Pipeline with Integrated Dead Code Analysis

Specialized pipeline for comprehensive code analysis including:
- Function call mapping
- Class interaction mapping
- Module dependency mapping
- Data flow analysis
- Interaction visualization
- Graph-based dead code analysis
- Real-time dead code detection
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

# Import interaction mappers
from mappers.map_code_interactions import CodeInteractionMapper
from mappers.enhanced_map_code_interactions import EnhancedCodeInteractionMapper

# Import analyzers
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.enhanced_dependency_analyzer import EnhancedDependencyAnalyzer
from analyzers.data_flow_analyzer import DataFlowAnalyzer
from analyzers.call_graph_analyzer import CallGraphAnalyzer
from analyzers.architecture_analyzer import ArchitectureAnalyzer

# Import visualizers
from visualizers.interaction_network import InteractionNetworkVisualizer
from visualizers.dependency_graph import DependencyGraphVisualizer
from visualizers.dashboard_generator import DashboardGenerator

# Import scripts
from scripts.extract_interactions import ExtractInteractions
from scripts.interaction_summary import InteractionSummary

# Import core components
from core.config import get_default_config
from plugins.plugin_registry import PluginRegistry
from plugins.plugin_manager import PluginManager

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

class EnhancedInteractionMappingPipeline:
    """Enhanced interaction mapping pipeline with integrated dead code analysis."""
    
    def __init__(self, project_root: str, config: Dict[str, Any] = None):
        self.project_root = Path(project_root)
        self.config = config or get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Initialize components
        self.code_mapper = CodeInteractionMapper(self.project_root, self.config)
        self.enhanced_mapper = EnhancedCodeInteractionMapper(self.project_root, self.config)
        self.dependency_analyzer = DependencyAnalyzer(self.project_root, self.config)
        self.enhanced_dependency_analyzer = EnhancedDependencyAnalyzer(self.project_root, self.config)
        self.data_flow_analyzer = DataFlowAnalyzer(self.project_root, self.config)
        self.call_graph_analyzer = CallGraphAnalyzer(self.project_root, self.config)
        self.architecture_analyzer = ArchitectureAnalyzer(self.project_root, self.config)
        
        # Initialize visualizers
        self.interaction_visualizer = InteractionNetworkVisualizer(self.project_root, self.config)
        self.dependency_visualizer = DependencyGraphVisualizer(self.project_root, self.config)
        self.dashboard_generator = DashboardGenerator(self.project_root, self.config)
        
        # Initialize scripts
        self.extract_interactions = ExtractInteractions(self.project_root, self.config)
        self.interaction_summary = InteractionSummary(self.project_root, self.config)
        
        # Initialize plugins
        self.plugin_registry = PluginRegistry()
        self.plugin_manager = PluginManager(self.plugin_registry)
        
        # Initialize dead code analyzer (will be set after interaction mapping)
        self.dead_code_analyzer = None
    
    def run_analysis(self, analysis_types: List[str] = None):
        """Run the enhanced interaction mapping analysis with dead code detection."""
        if analysis_types is None:
            analysis_types = ["all"]
        
        print("=" * 80)
        print("ENHANCED INTERACTION MAPPING PIPELINE WITH DEAD CODE ANALYSIS")
        print("=" * 80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Analysis types: {', '.join(analysis_types)}")
        print()
        
        start_time = time.time()
        
        # Run interaction mapping first
        self._run_interaction_mapping(analysis_types)
        
        # Initialize dead code analyzer with interaction data
        if self.results.get('basic_interaction_mapping'):
            self.dead_code_analyzer = GraphBasedDeadCodeAnalyzer(
                self.results['basic_interaction_mapping']
            )
            self.dead_code_analyzer.build_usage_graphs()
            
            # Run dead code analysis
            self._run_dead_code_analysis()
        
        # Generate enhanced reports
        self._generate_enhanced_reports()
        
        execution_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("ENHANCED INTERACTION MAPPING COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Reports saved to: {self.project_root / 'code_quality/reports/interaction_mapping'}")
        
        return self.results
    
    def _run_interaction_mapping(self, analysis_types: List[str]):
        """Run the standard interaction mapping analysis."""
        print("🔍 Running Interaction Mapping Analysis...")
        
        if "all" in analysis_types or "basic" in analysis_types:
            print("\n" + "="*60)
            print("Running Basic Code Interaction Mapping")
            print("="*60)
            self.results['basic_interaction_mapping'] = self.code_mapper.analyze()
        
        if "all" in analysis_types or "enhanced" in analysis_types:
            print("\n" + "="*60)
            print("Running Enhanced Code Interaction Mapping")
            print("="*60)
            self.results['enhanced_interaction_mapping'] = self.enhanced_mapper.analyze()
        
        if "all" in analysis_types or "dependency" in analysis_types:
            print("\n" + "="*60)
            print("Running Dependency Analysis")
            print("="*60)
            self.results['dependency_analysis'] = self.dependency_analyzer.analyze()
            self.results['enhanced_dependency_analysis'] = self.enhanced_dependency_analyzer.analyze()
        
        if "all" in analysis_types or "data_flow" in analysis_types:
            print("\n" + "="*60)
            print("Running Data Flow Analysis")
            print("="*60)
            self.results['data_flow_analysis'] = self.data_flow_analyzer.analyze()
        
        if "all" in analysis_types or "call_graph" in analysis_types:
            print("\n" + "="*60)
            print("Running Call Graph Analysis")
            print("="*60)
            self.results['call_graph_analysis'] = self.call_graph_analyzer.analyze()
        
        if "all" in analysis_types or "architecture" in analysis_types:
            print("\n" + "="*60)
            print("Running Architecture Analysis")
            print("="*60)
            self.results['architecture_analysis'] = self.architecture_analyzer.analyze()
    
    def _run_dead_code_analysis(self):
        """Run the integrated dead code analysis."""
        print("\n" + "="*60)
        print("Running Graph-Based Dead Code Analysis")
        print("="*60)
        
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
    
    def _generate_enhanced_reports(self):
        """Generate enhanced reports including dead code analysis."""
        print("\n" + "="*60)
        print("Generating Enhanced Reports")
        print("="*60)
        
        # Create reports directory
        reports_dir = self.project_root / "code_quality/reports/interaction_mapping"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual reports
        for analysis_type, results in self.results.items():
            report_file = reports_dir / f"{analysis_type}_{self.timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✅ Saved {analysis_type} report: {report_file}")
        
        # Generate dead code cleanup recommendations
        if 'dead_code_analysis' in self.results:
            self._generate_dead_code_cleanup_report()
        
        # Generate enhanced visualizations
        self._generate_enhanced_visualizations()
    
    def _generate_dead_code_cleanup_report(self):
        """Generate dead code cleanup recommendations."""
        dead_code_data = self.results['dead_code_analysis']
        
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
        reports_dir = self.project_root / "code_quality/reports/interaction_mapping"
        cleanup_file = reports_dir / f"dead_code_cleanup_recommendations_{self.timestamp}.json"
        with open(cleanup_file, 'w') as f:
            json.dump(cleanup_report, f, indent=2, default=str)
        
        print(f"✅ Generated dead code cleanup report: {cleanup_file}")
    
    def _generate_enhanced_visualizations(self):
        """Generate enhanced visualizations including dead code analysis."""
        try:
            # Generate interaction network with dead code highlighting
            if 'basic_interaction_mapping' in self.results:
                self.interaction_visualizer.generate_network_visualization(
                    self.results['basic_interaction_mapping'],
                    f"enhanced_interaction_network_{self.timestamp}.html"
                )
            
            # Generate dependency graph with dead code highlighting
            if 'dependency_analysis' in self.results:
                self.dependency_visualizer.generate_dependency_graph(
                    self.results['dependency_analysis'],
                    f"enhanced_dependency_graph_{self.timestamp}.html"
                )
            
            # Generate comprehensive dashboard
            self.dashboard_generator.generate_dashboard(
                self.results,
                f"enhanced_interaction_dashboard_{self.timestamp}.html"
            )
            
            print("✅ Generated enhanced visualizations")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

def main():
    """Main entry point for the enhanced interaction mapping pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Interaction Mapping Pipeline with Dead Code Analysis")
    parser.add_argument("--analysis-type", default="all", 
                       help="Type of analysis to run (all, basic, enhanced, dependency, data_flow, call_graph, architecture)")
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory")
    
    args = parser.parse_args()
    
    # Parse analysis types
    analysis_types = [t.strip() for t in args.analysis_type.split(",")]
    
    # Initialize and run pipeline
    pipeline = EnhancedInteractionMappingPipeline(args.project_root)
    results = pipeline.run_analysis(analysis_types)
    
    print(f"\n🎉 Enhanced interaction mapping pipeline completed successfully!")
    print(f"📊 Results: {len(results)} analysis types completed")
    
    return results

if __name__ == "__main__":
    main()
