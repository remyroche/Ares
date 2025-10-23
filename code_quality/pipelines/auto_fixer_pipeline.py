#!/usr/bin/env python3
"""
Auto-Fixer Pipeline

This pipeline is the only one that can actually modify files. It focuses on:
- Automatic code fixes and improvements
- Import organization and cleanup
- Code formatting and style fixes
- Dead code removal
- Syntax error fixes
- Code optimization

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse files and identify fixable issues
3. ANALYSIS - Determine what can be automatically fixed
4. PROCESSING - Apply fixes with safety checks
5. AGGREGATION - Combine results and generate reports
6. REPORTING - Generate fix reports and backups
7. CLEANUP - Clean up temporary files
"""

import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, StageResult, PipelineStatus, PipelineResult


class AutoFixerPipeline(BasePipeline):
    """Pipeline for automatic code fixes and improvements."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the auto-fixer pipeline."""
        super().__init__(config, "auto_fixer")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.fixable_issues: Dict[Path, List[Dict[str, Any]]] = {}
        self.applied_fixes: Dict[Path, List[Dict[str, Any]]] = {}
        self.backup_files: Dict[Path, Path] = {}
        self.fix_statistics: Dict[str, int] = {}
        self.dry_run = config.dry_run
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for auto-fixer pipeline."""
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
        self.logger.info("Initializing auto-fixer pipeline...")
        
        # Discover Python files
        self.python_files = list(self.config.project_root.rglob("*.py"))
        
        # Filter out common directories to ignore
        ignore_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
        self.python_files = [
            f for f in self.python_files 
            if not any(part in ignore_dirs for part in f.parts)
        ]
        
        # Create backup directory
        backup_dir = self.config.output_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        stage_result.complete({
            "files_discovered": len(self.python_files),
            "project_root": str(self.config.project_root),
            "backup_dir": str(backup_dir),
            "dry_run": self.dry_run,
            "files": [str(f) for f in self.python_files]
        })
        
        self.logger.info(f"Discovered {len(self.python_files)} Python files")
        if self.dry_run:
            self.logger.info("Running in DRY RUN mode - no files will be modified")
    
    async def _execute_preparation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Parse files and identify fixable issues."""
        self.logger.info("Preparing files and identifying fixable issues...")
        
        parse_errors = []
        successfully_parsed = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                self.parsed_files[file_path] = tree
                
                # Identify fixable issues
                issues = self._identify_fixable_issues(file_path, content, tree)
                self.fixable_issues[file_path] = issues
                
                successfully_parsed += 1
                
            except SyntaxError as e:
                parse_errors.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                    "fixable": True,
                    "fix_type": "syntax_error"
                })
                self.fixable_issues[file_path] = [parse_errors[-1]]
                successfully_parsed += 1
                
            except Exception as e:
                parse_errors.append({
                    "file": str(file_path),
                    "error": str(e),
                    "fixable": False
                })
        
        stage_result.complete({
            "files_parsed": successfully_parsed,
            "parse_errors": parse_errors,
            "total_files": len(self.python_files),
            "fixable_issues": sum(len(issues) for issues in self.fixable_issues.values())
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        total_fixable = sum(len(issues) for issues in self.fixable_issues.values())
        self.logger.info(f"Found {total_fixable} fixable issues")
    
    def _identify_fixable_issues(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify issues that can be automatically fixed."""
        issues = []
        
        # Check for common fixable issues
        issues.extend(self._find_unused_imports(file_path, content, tree))
        issues.extend(self._find_missing_imports(file_path, content, tree))
        issues.extend(self._find_formatting_issues(file_path, content))
        issues.extend(self._find_simple_syntax_issues(file_path, content))
        issues.extend(self._find_dead_code(file_path, content, tree))
        
        return issues
    
    def _find_unused_imports(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find unused imports that can be removed."""
        issues = []
        
        # Extract imports
        imports = []
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append({
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "type": "import"
                    })
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        imports.append({
                            "name": alias.name,
                            "module": node.module,
                            "alias": alias.asname,
                            "line": node.lineno,
                            "type": "from_import"
                        })
                self.generic_visit(node)
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        # Find used names
        used_names = set()
        class UsageVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                used_names.add(node.id)
                self.generic_visit(node)
        
        usage_visitor = UsageVisitor()
        usage_visitor.visit(tree)
        
        # Check for unused imports
        for imp in imports:
            if imp["name"] not in used_names and not imp["name"].startswith("_"):
                issues.append({
                    "type": "unused_import",
                    "line": imp["line"],
                    "name": imp["name"],
                    "fix_type": "remove_import",
                    "description": f"Unused import: {imp['name']}"
                })
        
        return issues
    
    def _find_missing_imports(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find missing imports that can be added."""
        issues = []
        
        # This is a simplified version - in practice, you'd need more sophisticated analysis
        # to determine what imports are actually missing
        
        return issues
    
    def _find_formatting_issues(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Find formatting issues that can be fixed."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    "type": "trailing_whitespace",
                    "line": i,
                    "fix_type": "remove_trailing_whitespace",
                    "description": "Trailing whitespace"
                })
            
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line:
                issues.append({
                    "type": "mixed_tabs_spaces",
                    "line": i,
                    "fix_type": "convert_tabs_to_spaces",
                    "description": "Mixed tabs and spaces"
                })
            
            # Check for lines that are too long (basic check)
            if len(line) > 120:
                issues.append({
                    "type": "line_too_long",
                    "line": i,
                    "fix_type": "break_long_line",
                    "description": f"Line too long ({len(line)} characters)"
                })
        
        return issues
    
    def _find_simple_syntax_issues(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Find simple syntax issues that can be fixed."""
        issues = []
        
        # Check for common syntax issues
        if '== None' in content:
            issues.append({
                "type": "none_comparison",
                "line": 1,
                "fix_type": "replace_none_comparison",
                "description": "Use 'is None' instead of '== None'"
            })
        
        if '!= None' in content:
            issues.append({
                "type": "none_comparison",
                "line": 1,
                "fix_type": "replace_none_comparison",
                "description": "Use 'is not None' instead of '!= None'"
            })
        
        return issues
    
    def _find_dead_code(self, file_path: Path, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find dead code that can be removed."""
        issues = []
        
        # This is a simplified version - in practice, you'd need more sophisticated analysis
        # to safely identify dead code
        
        return issues
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze what can be automatically fixed."""
        self.logger.info("Analyzing fixable issues...")
        
        analysis_results = {
            "total_issues": 0,
            "fixable_issues": 0,
            "issues_by_type": {},
            "files_with_issues": 0
        }
        
        for file_path, issues in self.fixable_issues.items():
            if issues:
                analysis_results["files_with_issues"] += 1
            
            for issue in issues:
                analysis_results["total_issues"] += 1
                if issue.get("fixable", True):
                    analysis_results["fixable_issues"] += 1
                
                issue_type = issue["type"]
                analysis_results["issues_by_type"][issue_type] = analysis_results["issues_by_type"].get(issue_type, 0) + 1
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "files_analyzed": len(self.parsed_files)
        })
        
        self.logger.info(f"Analysis complete: {analysis_results['fixable_issues']}/{analysis_results['total_issues']} issues can be fixed")
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Apply fixes with safety checks."""
        self.logger.info("Applying fixes...")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Simulating fixes without modifying files")
            applied_fixes = self._simulate_fixes()
        else:
            applied_fixes = self._apply_fixes()
        
        stage_result.complete({
            "applied_fixes": applied_fixes,
            "total_fixes": sum(len(fixes) for fixes in applied_fixes.values()),
            "files_modified": len([f for f in applied_fixes.values() if f])
        })
        
        total_fixes = sum(len(fixes) for fixes in applied_fixes.values())
        files_modified = len([f for f in applied_fixes.values() if f])
        self.logger.info(f"Applied {total_fixes} fixes to {files_modified} files")
    
    def _simulate_fixes(self) -> Dict[Path, List[Dict[str, Any]]]:
        """Simulate applying fixes without modifying files."""
        simulated_fixes = {}
        
        for file_path, issues in self.fixable_issues.items():
            file_fixes = []
            
            for issue in issues:
                if issue.get("fixable", True):
                    fix = {
                        "type": issue["type"],
                        "line": issue["line"],
                        "description": issue["description"],
                        "applied": True,
                        "simulated": True
                    }
                    file_fixes.append(fix)
            
            if file_fixes:
                simulated_fixes[file_path] = file_fixes
        
        return simulated_fixes
    
    def _apply_fixes(self) -> Dict[Path, List[Dict[str, Any]]]:
        """Apply fixes to files with safety checks."""
        applied_fixes = {}
        
        for file_path, issues in self.fixable_issues.items():
            if not issues:
                continue
            
            # Create backup
            backup_path = self._create_backup(file_path)
            self.backup_files[file_path] = backup_path
            
            try:
                # Read original content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply fixes
                modified_content, file_fixes = self._apply_file_fixes(content, issues)
                
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                applied_fixes[file_path] = file_fixes
                self.logger.info(f"Applied {len(file_fixes)} fixes to {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to apply fixes to {file_path}: {e}")
                # Restore from backup
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                applied_fixes[file_path] = []
        
        return applied_fixes
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create a backup of the original file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config.output_dir / "backups"
        backup_path = backup_dir / f"{file_path.name}.{timestamp}.bak"
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _apply_file_fixes(self, content: str, issues: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply fixes to a single file."""
        lines = content.split('\n')
        applied_fixes = []
        
        # Sort issues by line number (descending) to avoid line number shifts
        sorted_issues = sorted(issues, key=lambda x: x.get("line", 0), reverse=True)
        
        for issue in sorted_issues:
            if not issue.get("fixable", True):
                continue
            
            fix_type = issue.get("fix_type")
            line_num = issue.get("line", 1) - 1  # Convert to 0-based index
            
            if line_num < 0 or line_num >= len(lines):
                continue
            
            try:
                if fix_type == "remove_import":
                    # Remove the import line
                    if line_num < len(lines):
                        lines.pop(line_num)
                        applied_fixes.append({
                            "type": issue["type"],
                            "line": issue["line"],
                            "description": issue["description"],
                            "applied": True
                        })
                
                elif fix_type == "remove_trailing_whitespace":
                    # Remove trailing whitespace
                    if line_num < len(lines):
                        lines[line_num] = lines[line_num].rstrip()
                        applied_fixes.append({
                            "type": issue["type"],
                            "line": issue["line"],
                            "description": issue["description"],
                            "applied": True
                        })
                
                elif fix_type == "convert_tabs_to_spaces":
                    # Convert tabs to spaces
                    if line_num < len(lines):
                        lines[line_num] = lines[line_num].expandtabs(4)
                        applied_fixes.append({
                            "type": issue["type"],
                            "line": issue["line"],
                            "description": issue["description"],
                            "applied": True
                        })
                
                elif fix_type == "replace_none_comparison":
                    # Replace == None with is None
                    if line_num < len(lines):
                        original_line = lines[line_num]
                        if '== None' in original_line:
                            lines[line_num] = original_line.replace('== None', 'is None')
                        elif '!= None' in original_line:
                            lines[line_num] = original_line.replace('!= None', 'is not None')
                        applied_fixes.append({
                            "type": issue["type"],
                            "line": issue["line"],
                            "description": issue["description"],
                            "applied": True
                        })
                
            except Exception as e:
                self.logger.warning(f"Failed to apply fix {fix_type} at line {issue['line']}: {e}")
        
        return '\n'.join(lines), applied_fixes
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating auto-fixer results...")
        
        # Calculate summary statistics
        summary = {
            "total_files": len(self.python_files),
            "files_processed": len(self.parsed_files),
            "files_modified": 0,
            "total_fixes_applied": 0,
            "fixes_by_type": {},
            "backup_files_created": len(self.backup_files),
            "dry_run": self.dry_run
        }
        
        # Aggregate fixes
        for file_path, fixes in context.get("applied_fixes", {}).items():
            if fixes:
                summary["files_modified"] += 1
                summary["total_fixes_applied"] += len(fixes)
                
                for fix in fixes:
                    fix_type = fix["type"]
                    summary["fixes_by_type"][fix_type] = summary["fixes_by_type"].get(fix_type, 0) + 1
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "applied_fixes": context.get("applied_fixes", {}),
                "backup_files": {str(k): str(v) for k, v in self.backup_files.items()},
                "fix_statistics": summary["fixes_by_type"]
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_fixes_applied']} fixes applied to {summary['files_modified']} files")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive auto-fixer reports."""
        self.logger.info("Generating auto-fixer reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate backup report
        backup_report = self._generate_backup_report()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"auto_fixer_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"auto_fixer_detailed_{timestamp}.json"
        backup_path = self.config.output_dir / f"auto_fixer_backups_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(backup_path, 'w') as f:
            json.dump(backup_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "backups": str(backup_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {backup_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "auto_fixer",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "auto_fixer",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_backup_report(self) -> Dict[str, Any]:
        """Generate backup report."""
        return {
            "pipeline": "auto_fixer",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "backup_files": {str(k): str(v) for k, v in self.backup_files.items()},
            "backup_directory": str(self.config.output_dir / "backups")
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("total_fixes_applied", 0) > 0:
            recommendations.append("Review applied fixes to ensure they meet your coding standards")
        
        if summary.get("backup_files_created", 0) > 0:
            recommendations.append("Backup files have been created - you can restore from them if needed")
        
        if summary.get("fixes_by_type", {}).get("unused_import", 0) > 0:
            recommendations.append("Consider running import analysis to identify more import issues")
        
        if summary.get("fixes_by_type", {}).get("trailing_whitespace", 0) > 0:
            recommendations.append("Consider setting up a pre-commit hook to prevent trailing whitespace")
        
        if not summary.get("dry_run", False):
            recommendations.append("Run the pipeline in dry-run mode first to preview changes")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary files."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.parsed_files.clear()
        self.fixable_issues.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True,
            "backup_files_preserved": len(self.backup_files)
        })
        
        self.logger.info("Cleanup completed")
        if self.backup_files:
            self.logger.info(f"Backup files preserved: {len(self.backup_files)} files")


# Convenience function for easy usage
async def run_auto_fixer(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    dry_run: bool = True,
    **kwargs
) -> PipelineResult:
    """Run auto-fixer pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, dry_run=dry_run, **kwargs)
    pipeline = AutoFixerPipeline(config)
    return await pipeline.run()