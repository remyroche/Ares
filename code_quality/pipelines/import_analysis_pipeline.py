#!/usr/bin/env python3
"""
Import Analysis Pipeline

This pipeline focuses on comprehensive import analysis including:
- Import dependency mapping
- Circular dependency detection
- Unused import identification
- Import organization and optimization
- Module relationship analysis

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse imports and build dependency graph
3. ANALYSIS - Analyze import patterns and relationships
4. PROCESSING - Detect issues and optimize suggestions
5. AGGREGATION - Combine results and generate insights
6. REPORTING - Generate comprehensive import reports
7. CLEANUP - Clean up temporary data structures
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


class ImportAnalysisPipeline(BasePipeline):
    """Pipeline for comprehensive import analysis and optimization."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the import analysis pipeline."""
        super().__init__(config, "import_analysis")
        self.python_files: List[Path] = []
        self.module_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.import_data: Dict[str, Dict[str, Any]] = {}
        self.circular_dependencies: List[List[str]] = []
        self.unused_imports: Dict[str, List[Dict[str, Any]]] = {}
        self.import_issues: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for import analysis pipeline."""
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
        self.logger.info("Initializing import analysis pipeline...")
        
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
        """Parse imports and build dependency graph."""
        self.logger.info("Preparing import data and building dependency graph...")
        
        parse_errors = []
        successfully_parsed = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                
                # Extract module name
                module_name = self._get_module_name(file_path)
                
                # Extract imports
                imports = self._extract_imports(tree)
                
                # Store import data
                self.import_data[module_name] = {
                    "file_path": str(file_path),
                    "imports": imports,
                    "import_count": len(imports),
                    "external_imports": [imp for imp in imports if not imp.startswith('.')],
                    "internal_imports": [imp for imp in imports if imp.startswith('.')]
                }
                
                # Build dependency graph
                for import_name in imports:
                    self.module_graph[module_name].add(import_name)
                    self.reverse_graph[import_name].add(module_name)
                
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
            "modules_analyzed": len(self.import_data),
            "dependency_graph_size": len(self.module_graph)
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        self.logger.info(f"Built dependency graph with {len(self.module_graph)} modules")
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        # Get relative path from project root
        try:
            relative_path = file_path.relative_to(self.config.project_root)
        except ValueError:
            # File is outside project root
            return str(file_path)
        
        # Convert to module name
        parts = list(relative_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        
        return ".".join(parts)
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from an AST."""
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
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
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        return imports
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze import patterns and relationships."""
        self.logger.info("Analyzing import patterns and relationships...")
        
        analysis_results = {
            "circular_dependencies": 0,
            "unused_imports": 0,
            "import_issues": 0,
            "external_dependencies": 0,
            "internal_dependencies": 0
        }
        
        # Detect circular dependencies
        self.circular_dependencies = self._detect_circular_dependencies()
        analysis_results["circular_dependencies"] = len(self.circular_dependencies)
        
        # Find unused imports
        self.unused_imports = self._find_unused_imports()
        analysis_results["unused_imports"] = sum(len(imports) for imports in self.unused_imports.values())
        
        # Find import issues
        self.import_issues = self._find_import_issues()
        analysis_results["import_issues"] = sum(len(issues) for issues in self.import_issues.values())
        
        # Count external vs internal dependencies
        for module_data in self.import_data.values():
            analysis_results["external_dependencies"] += len(module_data["external_imports"])
            analysis_results["internal_dependencies"] += len(module_data["internal_imports"])
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "modules_analyzed": len(self.import_data)
        })
        
        self.logger.info(f"Analysis complete: {analysis_results['circular_dependencies']} circular dependencies, "
                        f"{analysis_results['unused_imports']} unused imports, "
                        f"{analysis_results['import_issues']} import issues")
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
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
                # Only consider internal modules
                if neighbor in self.import_data:
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for module in self.import_data.keys():
            if module not in visited:
                dfs(module, [])
        
        return circular_deps
    
    def _find_unused_imports(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find unused imports by analyzing usage."""
        unused_imports = {}
        
        for module_name, module_data in self.import_data.items():
            file_path = Path(module_data["file_path"])
            imports = module_data["imports"]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                # Find all name usages
                used_names = set()
                class UsageVisitor(ast.NodeVisitor):
                    def visit_Name(self, node):
                        used_names.add(node.id)
                        self.generic_visit(node)
                    
                    def visit_Attribute(self, node):
                        # Handle attribute access like module.function
                        if isinstance(node.value, ast.Name):
                            used_names.add(node.value.id)
                        self.generic_visit(node)
                
                visitor = UsageVisitor()
                visitor.visit(tree)
                
                # Check which imports are unused
                module_unused = []
                for import_name in imports:
                    # Extract the main module name
                    main_module = import_name.split('.')[0]
                    if main_module not in used_names and not main_module.startswith('_'):
                        module_unused.append({
                            "import": import_name,
                            "reason": "not_used"
                        })
                
                if module_unused:
                    unused_imports[module_name] = module_unused
                    
            except Exception as e:
                self.logger.warning(f"Could not analyze unused imports for {module_name}: {e}")
        
        return unused_imports
    
    def _find_import_issues(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find various import-related issues."""
        import_issues = {}
        
        for module_name, module_data in self.import_data.items():
            issues = []
            imports = module_data["imports"]
            
            # Check for duplicate imports
            seen_imports = set()
            for import_name in imports:
                if import_name in seen_imports:
                    issues.append({
                        "type": "duplicate_import",
                        "import": import_name,
                        "message": f"Duplicate import: {import_name}"
                    })
                seen_imports.add(import_name)
            
            # Check for wildcard imports
            for import_name in imports:
                if import_name.endswith('.*'):
                    issues.append({
                        "type": "wildcard_import",
                        "import": import_name,
                        "message": f"Wildcard import: {import_name}"
                    })
            
            # Check for very long import chains
            for import_name in imports:
                if len(import_name.split('.')) > 4:
                    issues.append({
                        "type": "deep_import",
                        "import": import_name,
                        "message": f"Deep import chain: {import_name}"
                    })
            
            if issues:
                import_issues[module_name] = issues
        
        return import_issues
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process and categorize all import issues."""
        self.logger.info("Processing import issues and generating optimization suggestions...")
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions()
        
        # Categorize issues by severity
        issue_categories = {
            "critical": [],  # Circular dependencies
            "high": [],      # Unused imports
            "medium": [],    # Import issues
            "low": []        # Optimization suggestions
        }
        
        # Add circular dependencies as critical
        for cycle in self.circular_dependencies:
            issue_categories["critical"].append({
                "type": "circular_dependency",
                "cycle": cycle,
                "message": f"Circular dependency: {' -> '.join(cycle)}"
            })
        
        # Add unused imports as high priority
        for module, unused in self.unused_imports.items():
            for unused_import in unused:
                issue_categories["high"].append({
                    "type": "unused_import",
                    "module": module,
                    "import": unused_import["import"],
                    "message": f"Unused import in {module}: {unused_import['import']}"
                })
        
        # Add import issues as medium priority
        for module, issues in self.import_issues.items():
            for issue in issues:
                issue_categories["medium"].append({
                    "type": issue["type"],
                    "module": module,
                    "import": issue["import"],
                    "message": issue["message"]
                })
        
        stage_result.complete({
            "issue_categories": issue_categories,
            "optimization_suggestions": optimization_suggestions,
            "total_issues": sum(len(issues) for issues in issue_categories.values())
        })
        
        total_issues = sum(len(issues) for issues in issue_categories.values())
        self.logger.info(f"Processed {total_issues} import issues across all categories")
    
    def _generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for imports."""
        suggestions = []
        
        # Suggest import organization
        for module_name, module_data in self.import_data.items():
            imports = module_data["imports"]
            if len(imports) > 10:
                suggestions.append({
                    "type": "import_organization",
                    "module": module_name,
                    "message": f"Module {module_name} has {len(imports)} imports - consider organizing into groups"
                })
        
        # Suggest dependency reduction
        high_dependency_modules = [
            (module, len(deps)) for module, deps in self.module_graph.items()
            if len(deps) > 15
        ]
        for module, dep_count in high_dependency_modules:
            suggestions.append({
                "type": "dependency_reduction",
                "module": module,
                "message": f"Module {module} has {dep_count} dependencies - consider reducing coupling"
            })
        
        return suggestions
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating import analysis results...")
        
        # Calculate summary statistics
        summary = {
            "total_modules": len(self.import_data),
            "total_imports": sum(len(data["imports"]) for data in self.import_data.values()),
            "external_imports": sum(len(data["external_imports"]) for data in self.import_data.values()),
            "internal_imports": sum(len(data["internal_imports"]) for data in self.import_data.values()),
            "circular_dependencies": len(self.circular_dependencies),
            "unused_imports": sum(len(imports) for imports in self.unused_imports.values()),
            "import_issues": sum(len(issues) for issues in self.import_issues.values()),
            "dependency_graph_stats": {
                "nodes": len(self.module_graph),
                "edges": sum(len(deps) for deps in self.module_graph.values()),
                "max_dependencies": max(len(deps) for deps in self.module_graph.values()) if self.module_graph else 0
            }
        }
        
        # Find most/least dependent modules
        dependency_counts = [(module, len(deps)) for module, deps in self.module_graph.items()]
        dependency_counts.sort(key=lambda x: x[1], reverse=True)
        
        summary["most_dependent_modules"] = dependency_counts[:5]
        summary["least_dependent_modules"] = dependency_counts[-5:] if len(dependency_counts) > 5 else dependency_counts
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "import_data": self.import_data,
                "circular_dependencies": self.circular_dependencies,
                "unused_imports": self.unused_imports,
                "import_issues": self.import_issues,
                "module_graph": dict(self.module_graph)
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_imports']} total imports, "
                        f"{summary['circular_dependencies']} circular dependencies")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive import analysis reports."""
        self.logger.info("Generating import analysis reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate dependency graph report
        graph_report = self._generate_dependency_graph_report()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"import_analysis_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"import_analysis_detailed_{timestamp}.json"
        graph_path = self.config.output_dir / f"import_analysis_graph_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(graph_path, 'w') as f:
            json.dump(graph_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "dependency_graph": str(graph_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {graph_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_dependency_graph_report(self) -> Dict[str, Any]:
        """Generate dependency graph report."""
        return {
            "pipeline": "import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "dependency_graph": {
                "nodes": list(self.module_graph.keys()),
                "edges": [(source, target) for source, targets in self.module_graph.items() for target in targets],
                "circular_dependencies": self.circular_dependencies
            }
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("circular_dependencies", 0) > 0:
            recommendations.append("Resolve circular dependencies to improve code maintainability")
        
        if summary.get("unused_imports", 0) > 0:
            recommendations.append("Remove unused imports to reduce code bloat")
        
        if summary.get("import_issues", 0) > 0:
            recommendations.append("Fix import issues (duplicates, wildcards, deep chains)")
        
        if summary.get("dependency_graph_stats", {}).get("max_dependencies", 0) > 20:
            recommendations.append("Consider refactoring modules with high dependency counts")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary data structures."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.module_graph.clear()
        self.reverse_graph.clear()
        self.import_data.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_import_analysis(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run import analysis pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = ImportAnalysisPipeline(config)
    return await pipeline.run()