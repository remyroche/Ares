#!/usr/bin/env python3
"""
Code Graph/Mapping Pipeline

This pipeline focuses on building comprehensive code graphs and mappings including:
- Call graph construction
- Dependency mapping
- Module relationship analysis
- Code flow analysis
- Architecture visualization
- Component interaction mapping

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse files and extract relationships
3. ANALYSIS - Build comprehensive code graphs
4. PROCESSING - Analyze graph properties and patterns
5. AGGREGATION - Combine results and generate insights
6. REPORTING - Generate graph reports and visualizations
7. CLEANUP - Clean up temporary structures
"""

import ast
import json
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, StageResult, PipelineStatus, PipelineResult


class CodeGraphPipeline(BasePipeline):
    """Pipeline for comprehensive code graph construction and analysis."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the code graph pipeline."""
        super().__init__(config, "code_graph")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.module_graph: Dict[str, Set[str]] = defaultdict(set)
        self.class_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self.function_calls: Dict[str, List[Dict[str, Any]]] = {}
        self.module_relationships: Dict[str, Dict[str, Any]] = {}
        self.architecture_components: Dict[str, List[str]] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for code graph pipeline."""
        return [
            PipelineStage.INITIALIZATION,
            PipelineStage.PREPARATION,
            PipelineStage.ANALYSIS,
            PipelineStage.PROCESSING,
            PipelineStage.AGGREGATION,
            PipelineStage.REPORTING,
            PipelineStage.CLEANUP
        ]
    
    async def execute_stage(self, stage: PipelineStage, context: Dict[str, Any]) -> StageResult:
        """Execute a specific pipeline stage."""
        stage_result = StageResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            if stage == PipelineStage.INITIALIZATION:
                await self._execute_initialization(stage_result, context)
            elif stage == PipelineStage.PREPARATION:
                await self._execute_preparation(stage_result, context)
            elif stage == PipelineStage.ANALYSIS:
                await self._execute_analysis(stage_result, context)
            elif stage == PipelineStage.PROCESSING:
                await self._execute_processing(stage_result, context)
            elif stage == PipelineStage.AGGREGATION:
                await self._execute_aggregation(stage_result, context)
            elif stage == PipelineStage.REPORTING:
                await self._execute_reporting(stage_result, context)
            elif stage == PipelineStage.CLEANUP:
                await self._execute_cleanup(stage_result, context)
            
            return stage_result
            
        except Exception as e:
            stage_result.fail([f"Stage {stage.value} failed: {e}"])
            return stage_result
    
    async def _execute_initialization(self, stage_result: StageResult, context: Dict[str, Any]):
        """Initialize the pipeline and discover Python files."""
        self.logger.info("Initializing code graph pipeline...")
        
        # Discover Python files
        self.python_files = list(self.config.project_root.rglob("*.py"))
        
        # Filter out common directories to ignore
        ignore_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
        self.python_files = [
            f for f in self.python_files 
            if not any(part in ignore_dirs for part in f.parts)
        ]
        
        stage_result.complete({
            "files_discovered": len(self.python_files),
            "project_root": str(self.config.project_root),
            "files": [str(f) for f in self.python_files]
        })
        
        self.logger.info(f"Discovered {len(self.python_files)} Python files")
    
    async def _execute_preparation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Parse files and extract basic relationships."""
        self.logger.info("Preparing files and extracting relationships...")
        
        parse_errors = []
        successfully_parsed = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                self.parsed_files[file_path] = tree
                
                # Extract module name
                module_name = self._get_module_name(file_path)
                
                # Extract basic relationships
                self._extract_relationships(file_path, tree, module_name)
                
                successfully_parsed += 1
                
            except SyntaxError as e:
                parse_errors.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg
                })
            except Exception as e:
                parse_errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        stage_result.complete({
            "files_parsed": successfully_parsed,
            "parse_errors": parse_errors,
            "total_files": len(self.python_files),
            "modules_analyzed": len(self.module_graph)
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        self.logger.info(f"Extracted relationships for {len(self.module_graph)} modules")
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(self.config.project_root)
        except ValueError:
            return str(file_path)
        
        parts = list(relative_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        
        return ".".join(parts)
    
    def _extract_relationships(self, file_path: Path, tree: ast.AST, module_name: str):
        """Extract relationships from a parsed file."""
        # Extract imports and dependencies
        imports = []
        function_calls = []
        class_definitions = []
        
        class RelationshipVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append(alias.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        if alias.name == "*":
                            imports.append(f"{node.module}.*")
                        else:
                            imports.append(f"{node.module}.{alias.name}")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    function_calls.append({
                        "function": node.func.id,
                        "line": node.lineno,
                        "type": "direct_call"
                    })
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        function_calls.append({
                            "function": f"{node.func.value.id}.{node.func.attr}",
                            "line": node.lineno,
                            "type": "method_call"
                        })
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                class_definitions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                })
                self.generic_visit(node)
        
        visitor = RelationshipVisitor()
        visitor.visit(tree)
        
        # Store relationships
        self.dependency_graph[module_name] = set(imports)
        self.function_calls[module_name] = function_calls
        
        # Build module graph
        for import_name in imports:
            self.module_graph[module_name].add(import_name)
        
        # Build class hierarchy
        for class_def in class_definitions:
            for base in class_def["bases"]:
                self.class_hierarchy[class_def["name"]].add(base)
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Build comprehensive code graphs."""
        self.logger.info("Building comprehensive code graphs...")
        
        analysis_results = {
            "call_graph_nodes": 0,
            "dependency_edges": 0,
            "module_relationships": 0,
            "class_hierarchies": 0
        }
        
        # Build call graph
        self._build_call_graph()
        analysis_results["call_graph_nodes"] = len(self.call_graph)
        
        # Analyze module relationships
        self._analyze_module_relationships()
        analysis_results["module_relationships"] = len(self.module_relationships)
        
        # Identify architecture components
        self._identify_architecture_components()
        
        # Calculate dependency metrics
        analysis_results["dependency_edges"] = sum(len(deps) for deps in self.dependency_graph.values())
        analysis_results["class_hierarchies"] = len(self.class_hierarchy)
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "modules_analyzed": len(self.parsed_files)
        })
        
        self.logger.info(f"Analysis complete: {analysis_results['call_graph_nodes']} call graph nodes, "
                        f"{analysis_results['dependency_edges']} dependency edges")
    
    def _build_call_graph(self):
        """Build comprehensive call graph."""
        for module_name, calls in self.function_calls.items():
            for call in calls:
                function_name = call["function"]
                self.call_graph[module_name].add(function_name)
                
                # Add cross-module calls
                if "." in function_name:
                    parts = function_name.split(".")
                    if len(parts) >= 2:
                        target_module = parts[0]
                        target_function = ".".join(parts[1:])
                        self.call_graph[f"{module_name}->{target_module}"].add(target_function)
    
    def _analyze_module_relationships(self):
        """Analyze relationships between modules."""
        for module_name, dependencies in self.dependency_graph.items():
            relationships = {
                "dependencies": list(dependencies),
                "dependents": [],
                "coupling_score": len(dependencies),
                "cohesion_score": 0
            }
            
            # Find modules that depend on this one
            for other_module, other_deps in self.dependency_graph.items():
                if other_module != module_name and module_name in other_deps:
                    relationships["dependents"].append(other_module)
            
            # Calculate cohesion score (simplified)
            relationships["cohesion_score"] = len(relationships["dependents"])
            
            self.module_relationships[module_name] = relationships
    
    def _identify_architecture_components(self):
        """Identify architectural components and patterns."""
        # Group modules by directory structure
        components = defaultdict(list)
        
        for module_name in self.module_graph.keys():
            parts = module_name.split(".")
            if len(parts) > 1:
                component = parts[0]
                components[component].append(module_name)
            else:
                components["root"].append(module_name)
        
        self.architecture_components = dict(components)
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process graph data and identify patterns."""
        self.logger.info("Processing graph data and identifying patterns...")
        
        # Identify graph patterns
        patterns = {
            "circular_dependencies": self._find_circular_dependencies(),
            "high_coupling_modules": self._find_high_coupling_modules(),
            "isolated_modules": self._find_isolated_modules(),
            "hub_modules": self._find_hub_modules(),
            "layered_architecture": self._identify_layered_architecture()
        }
        
        # Calculate graph metrics
        metrics = {
            "total_nodes": len(self.module_graph),
            "total_edges": sum(len(deps) for deps in self.module_graph.values()),
            "average_degree": 0,
            "max_degree": 0,
            "min_degree": float('inf'),
            "graph_density": 0
        }
        
        if self.module_graph:
            degrees = [len(deps) for deps in self.module_graph.values()]
            metrics["average_degree"] = sum(degrees) / len(degrees)
            metrics["max_degree"] = max(degrees)
            metrics["min_degree"] = min(degrees)
            
            # Calculate graph density
            n = len(self.module_graph)
            max_edges = n * (n - 1)  # Directed graph
            if max_edges > 0:
                metrics["graph_density"] = metrics["total_edges"] / max_edges
        
        stage_result.complete({
            "patterns": patterns,
            "metrics": metrics,
            "total_patterns": sum(len(pattern) for pattern in patterns.values())
        })
        
        total_patterns = sum(len(pattern) for pattern in patterns.values())
        self.logger.info(f"Processed graph data: {total_patterns} patterns identified")
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies using DFS."""
        circular_deps = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular_deps.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.module_graph.get(node, []):
                if neighbor in self.module_graph:
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for module in self.module_graph.keys():
            if module not in visited:
                dfs(module, [])
        
        return circular_deps
    
    def _find_high_coupling_modules(self) -> List[Dict[str, Any]]:
        """Find modules with high coupling."""
        high_coupling = []
        
        for module, relationships in self.module_relationships.items():
            coupling_score = relationships["coupling_score"]
            if coupling_score > 10:  # Threshold for high coupling
                high_coupling.append({
                    "module": module,
                    "coupling_score": coupling_score,
                    "dependencies": relationships["dependencies"]
                })
        
        return sorted(high_coupling, key=lambda x: x["coupling_score"], reverse=True)
    
    def _find_isolated_modules(self) -> List[str]:
        """Find isolated modules (no dependencies or dependents)."""
        isolated = []
        
        for module, relationships in self.module_relationships.items():
            if (len(relationships["dependencies"]) == 0 and 
                len(relationships["dependents"]) == 0):
                isolated.append(module)
        
        return isolated
    
    def _find_hub_modules(self) -> List[Dict[str, Any]]:
        """Find hub modules (highly connected)."""
        hubs = []
        
        for module, relationships in self.module_relationships.items():
            total_connections = (len(relationships["dependencies"]) + 
                               len(relationships["dependents"]))
            if total_connections > 15:  # Threshold for hub
                hubs.append({
                    "module": module,
                    "total_connections": total_connections,
                    "dependencies": len(relationships["dependencies"]),
                    "dependents": len(relationships["dependents"])
                })
        
        return sorted(hubs, key=lambda x: x["total_connections"], reverse=True)
    
    def _identify_layered_architecture(self) -> Dict[str, List[str]]:
        """Identify layered architecture patterns."""
        layers = {
            "presentation": [],
            "business": [],
            "data": [],
            "infrastructure": []
        }
        
        # Simple heuristic based on module names
        for module in self.module_graph.keys():
            module_lower = module.lower()
            if any(keyword in module_lower for keyword in ["ui", "gui", "view", "controller"]):
                layers["presentation"].append(module)
            elif any(keyword in module_lower for keyword in ["service", "business", "logic", "model"]):
                layers["business"].append(module)
            elif any(keyword in module_lower for keyword in ["data", "db", "repository", "dao"]):
                layers["data"].append(module)
            elif any(keyword in module_lower for keyword in ["config", "util", "helper", "common"]):
                layers["infrastructure"].append(module)
        
        return layers
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating code graph results...")
        
        # Calculate summary statistics
        summary = {
            "total_modules": len(self.module_graph),
            "total_dependencies": sum(len(deps) for deps in self.dependency_graph.values()),
            "total_function_calls": sum(len(calls) for calls in self.function_calls.values()),
            "architecture_components": len(self.architecture_components),
            "graph_metrics": context.get("metrics", {}),
            "patterns_found": {
                "circular_dependencies": len(context.get("patterns", {}).get("circular_dependencies", [])),
                "high_coupling_modules": len(context.get("patterns", {}).get("high_coupling_modules", [])),
                "isolated_modules": len(context.get("patterns", {}).get("isolated_modules", [])),
                "hub_modules": len(context.get("patterns", {}).get("hub_modules", []))
            }
        }
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "call_graph": dict(self.call_graph),
                "dependency_graph": dict(self.dependency_graph),
                "module_graph": dict(self.module_graph),
                "class_hierarchy": dict(self.class_hierarchy),
                "module_relationships": self.module_relationships,
                "architecture_components": self.architecture_components
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_modules']} modules, "
                        f"{summary['total_dependencies']} dependencies")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive code graph reports."""
        self.logger.info("Generating code graph reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate graph visualization data
        graph_data = self._generate_graph_visualization_data()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"code_graph_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"code_graph_detailed_{timestamp}.json"
        graph_path = self.config.output_dir / f"code_graph_visualization_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "graph_visualization": str(graph_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {graph_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "code_graph",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "code_graph",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_graph_visualization_data(self) -> Dict[str, Any]:
        """Generate data for graph visualization."""
        nodes = []
        edges = []
        
        # Create nodes
        for module in self.module_graph.keys():
            relationships = self.module_relationships.get(module, {})
            nodes.append({
                "id": module,
                "label": module,
                "dependencies": len(relationships.get("dependencies", [])),
                "dependents": len(relationships.get("dependents", [])),
                "coupling_score": relationships.get("coupling_score", 0)
            })
        
        # Create edges
        for module, dependencies in self.module_graph.items():
            for dep in dependencies:
                edges.append({
                    "source": module,
                    "target": dep,
                    "type": "dependency"
                })
        
        return {
            "pipeline": "code_graph",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "graph": {
                "nodes": nodes,
                "edges": edges
            }
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("patterns_found", {}).get("circular_dependencies", 0) > 0:
            recommendations.append("Resolve circular dependencies to improve maintainability")
        
        if summary.get("patterns_found", {}).get("high_coupling_modules", 0) > 0:
            recommendations.append("Reduce coupling in highly connected modules")
        
        if summary.get("patterns_found", {}).get("isolated_modules", 0) > 0:
            recommendations.append("Review isolated modules - they might be unused or need integration")
        
        if summary.get("patterns_found", {}).get("hub_modules", 0) > 0:
            recommendations.append("Consider refactoring hub modules to reduce complexity")
        
        if summary.get("graph_metrics", {}).get("graph_density", 0) > 0.5:
            recommendations.append("High graph density suggests tight coupling - consider modularization")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary data structures."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.parsed_files.clear()
        self.call_graph.clear()
        self.dependency_graph.clear()
        self.module_graph.clear()
        self.class_hierarchy.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_code_graph_analysis(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run code graph analysis pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = CodeGraphPipeline(config)
    return await pipeline.run()