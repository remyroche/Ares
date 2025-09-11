#!/usr/bin/env python3
"""
Syntax Validation Pipeline

This pipeline focuses on code quality related to syntax, imports, function parameters,
and validation. It provides comprehensive syntax analysis without modifying files.

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse and validate file structure
3. ANALYSIS - Syntax validation, import analysis, parameter validation
4. PROCESSING - Aggregate and categorize issues
5. AGGREGATION - Combine results from all files
6. REPORTING - Generate comprehensive reports
7. CLEANUP - Clean up temporary files
"""

import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, StageResult, PipelineStatus, PipelineResult


class SyntaxValidationPipeline(BasePipeline):
    """Pipeline for comprehensive syntax validation and code quality analysis."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the syntax validation pipeline."""
        super().__init__(config, "syntax_validation")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.syntax_issues: Dict[Path, List[Dict[str, Any]]] = {}
        self.import_issues: Dict[Path, List[Dict[str, Any]]] = {}
        self.parameter_issues: Dict[Path, List[Dict[str, Any]]] = {}
        self.validation_issues: Dict[Path, List[Dict[str, Any]]] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for syntax validation pipeline."""
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
            start_time=stage_result.start_time if 'stage_result' in locals() else None
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
        self.logger.info("Initializing syntax validation pipeline...")
        
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
        """Parse Python files and validate basic structure."""
        self.logger.info("Preparing files for analysis...")
        
        parse_errors = []
        successfully_parsed = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                self.parsed_files[file_path] = tree
                successfully_parsed += 1
                
            except SyntaxError as e:
                parse_errors.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                    "text": e.text
                })
            except Exception as e:
                parse_errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        stage_result.complete({
            "files_parsed": successfully_parsed,
            "parse_errors": parse_errors,
            "total_files": len(self.python_files)
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        if parse_errors:
            self.logger.warning(f"Found {len(parse_errors)} parse errors")
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Perform comprehensive syntax and code quality analysis."""
        self.logger.info("Analyzing syntax and code quality...")
        
        analysis_results = {
            "syntax_issues": 0,
            "import_issues": 0,
            "parameter_issues": 0,
            "validation_issues": 0
        }
        
        for file_path, tree in self.parsed_files.items():
            # Analyze syntax issues
            syntax_issues = self._analyze_syntax_issues(file_path, tree)
            self.syntax_issues[file_path] = syntax_issues
            analysis_results["syntax_issues"] += len(syntax_issues)
            
            # Analyze import issues
            import_issues = self._analyze_import_issues(file_path, tree)
            self.import_issues[file_path] = import_issues
            analysis_results["import_issues"] += len(import_issues)
            
            # Analyze parameter issues
            parameter_issues = self._analyze_parameter_issues(file_path, tree)
            self.parameter_issues[file_path] = parameter_issues
            analysis_results["parameter_issues"] += len(parameter_issues)
            
            # Analyze validation issues
            validation_issues = self._analyze_validation_issues(file_path, tree)
            self.validation_issues[file_path] = validation_issues
            analysis_results["validation_issues"] += len(validation_issues)
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "files_analyzed": len(self.parsed_files)
        })
        
        total_issues = sum(analysis_results.values())
        self.logger.info(f"Analysis complete: {total_issues} issues found across {len(self.parsed_files)} files")
    
    def _analyze_syntax_issues(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze syntax-related issues."""
        issues = []
        
        class SyntaxVisitor(ast.NodeVisitor):
            def visit_Compare(self, node):
                # Check for dangerous comparisons
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    if any(isinstance(op, ast.Is) or isinstance(op, ast.IsNot) for op in node.ops):
                        issues.append({
                            "type": "dangerous_string_comparison",
                            "line": node.lineno,
                            "message": "Using 'is' or 'is not' with string literals is dangerous"
                        })
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Check for unused assignments
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                    if target_name.startswith('_'):
                        issues.append({
                            "type": "unused_assignment",
                            "line": node.lineno,
                            "message": f"Assignment to '{target_name}' may be unused"
                        })
                self.generic_visit(node)
            
            def visit_For(self, node):
                # Check for unused loop variables
                if isinstance(node.target, ast.Name) and node.target.id.startswith('_'):
                    issues.append({
                        "type": "unused_loop_variable",
                        "line": node.lineno,
                        "message": f"Loop variable '{node.target.id}' may be unused"
                    })
                self.generic_visit(node)
        
        visitor = SyntaxVisitor()
        visitor.visit(tree)
        return issues
    
    def _analyze_import_issues(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze import-related issues."""
        issues = []
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": node.module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
                self.generic_visit(node)
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        # Check for common import issues
        import_names = [imp["name"] for imp in imports]
        
        # Check for duplicate imports
        seen_imports = set()
        for imp in imports:
            key = (imp["name"], imp.get("module"))
            if key in seen_imports:
                issues.append({
                    "type": "duplicate_import",
                    "line": imp["line"],
                    "message": f"Duplicate import: {imp['name']}"
                })
            seen_imports.add(key)
        
        # Check for unused imports (basic check)
        used_names = set()
        class UsageVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                used_names.add(node.id)
                self.generic_visit(node)
        
        usage_visitor = UsageVisitor()
        usage_visitor.visit(tree)
        
        for imp in imports:
            if imp["name"] not in used_names and not imp["name"].startswith("_"):
                issues.append({
                    "type": "unused_import",
                    "line": imp["line"],
                    "message": f"Unused import: {imp['name']}"
                })
        
        return issues
    
    def _analyze_parameter_issues(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze function parameter issues."""
        issues = []
        
        class ParameterVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for too many parameters
                if len(node.args.args) > 7:
                    issues.append({
                        "type": "too_many_parameters",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' has {len(node.args.args)} parameters (consider refactoring)"
                    })
                
                # Check for missing type hints
                if not node.returns and len(node.args.args) > 0:
                    issues.append({
                        "type": "missing_type_hints",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' missing type hints"
                    })
                
                # Check for mutable default arguments
                for arg in node.args.defaults:
                    if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "type": "mutable_default_argument",
                            "line": node.lineno,
                            "message": f"Function '{node.name}' has mutable default argument"
                        })
                
                self.generic_visit(node)
        
        visitor = ParameterVisitor()
        visitor.visit(tree)
        return issues
    
    def _analyze_validation_issues(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze validation and error handling issues."""
        issues = []
        
        class ValidationVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for functions without docstrings
                if not ast.get_docstring(node):
                    issues.append({
                        "type": "missing_docstring",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' missing docstring"
                    })
                
                # Check for bare except clauses
                for child in ast.walk(node):
                    if isinstance(child, ast.ExceptHandler) and child.type is None:
                        issues.append({
                            "type": "bare_except",
                            "line": child.lineno,
                            "message": "Bare except clause (should specify exception type)"
                        })
                
                self.generic_visit(node)
        
        visitor = ValidationVisitor()
        visitor.visit(tree)
        return issues
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process and categorize all issues."""
        self.logger.info("Processing and categorizing issues...")
        
        # Combine all issues by type
        all_issues = {
            "syntax": [],
            "imports": [],
            "parameters": [],
            "validation": []
        }
        
        for file_path in self.parsed_files.keys():
            file_str = str(file_path)
            
            # Add syntax issues
            for issue in self.syntax_issues.get(file_path, []):
                all_issues["syntax"].append({
                    "file": file_str,
                    **issue
                })
            
            # Add import issues
            for issue in self.import_issues.get(file_path, []):
                all_issues["imports"].append({
                    "file": file_str,
                    **issue
                })
            
            # Add parameter issues
            for issue in self.parameter_issues.get(file_path, []):
                all_issues["parameters"].append({
                    "file": file_str,
                    **issue
                })
            
            # Add validation issues
            for issue in self.validation_issues.get(file_path, []):
                all_issues["validation"].append({
                    "file": file_str,
                    **issue
                })
        
        stage_result.complete({
            "processed_issues": all_issues,
            "total_issues": sum(len(issues) for issues in all_issues.values())
        })
        
        total_issues = sum(len(issues) for issues in all_issues.values())
        self.logger.info(f"Processed {total_issues} issues across all categories")
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating results...")
        
        # Calculate summary statistics
        summary = {
            "total_files": len(self.python_files),
            "parsed_files": len(self.parsed_files),
            "failed_parses": len(self.python_files) - len(self.parsed_files),
            "total_issues": 0,
            "issues_by_type": {},
            "issues_by_file": {},
            "severity_distribution": {"low": 0, "medium": 0, "high": 0}
        }
        
        # Aggregate issues by type
        for category, issues in context.get("processed_issues", {}).items():
            summary["issues_by_type"][category] = len(issues)
            summary["total_issues"] += len(issues)
        
        # Aggregate issues by file
        for file_path in self.parsed_files.keys():
            file_issues = (
                len(self.syntax_issues.get(file_path, [])) +
                len(self.import_issues.get(file_path, [])) +
                len(self.parameter_issues.get(file_path, [])) +
                len(self.validation_issues.get(file_path, []))
            )
            summary["issues_by_file"][str(file_path)] = file_issues
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "syntax_issues": self.syntax_issues,
                "import_issues": self.import_issues,
                "parameter_issues": self.parameter_issues,
                "validation_issues": self.validation_issues
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_issues']} total issues found")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive reports."""
        self.logger.info("Generating reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"syntax_validation_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"syntax_validation_detailed_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "syntax_validation",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "syntax_validation",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_issues": aggregated_data
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("failed_parses", 0) > 0:
            recommendations.append("Fix syntax errors in files that failed to parse")
        
        if summary.get("issues_by_type", {}).get("imports", 0) > 0:
            recommendations.append("Review and clean up import statements")
        
        if summary.get("issues_by_type", {}).get("parameters", 0) > 0:
            recommendations.append("Add type hints and review function parameters")
        
        if summary.get("issues_by_type", {}).get("validation", 0) > 0:
            recommendations.append("Add docstrings and improve error handling")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary files and resources."""
        self.logger.info("Cleaning up...")
        
        # Clear parsed files from memory
        self.parsed_files.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_syntax_validation(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run syntax validation pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = SyntaxValidationPipeline(config)
    return await pipeline.run()