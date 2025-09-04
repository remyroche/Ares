"""
Dead Code Auto-Fixer Plugin for Pipeline Integration

This plugin automatically fixes high-confidence dead code issues, particularly
unused imports, while preserving all functional code and public APIs.

Features:
- High-confidence unused import removal
- Conservative approach with dry-run support
- Integration with the plugin system
- Comprehensive reporting and rollback capabilities
"""

import ast
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Simple base plugin class for standalone usage
class BasePlugin:
    def __init__(self):
        self.name = ""
        self.description = ""
        self.category = ""
        self.priority = ""
        self.version = ""
        self.logger = None
    
    def configure(self, config: Dict[str, Any]) -> None:
        pass
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": False, "error": "Not implemented"}

# Import the analyzer classes
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analyzers"))
from improved_dead_code_analyzer import ImprovedDeadCodeAnalyzer, DeadCodeIssue


@dataclass
class FixResult:
    """Result of a single file fix operation."""
    file_path: str
    success: bool
    issues_fixed: int
    changes_made: List[Dict[str, Any]]
    error_message: Optional[str] = None
    backup_path: Optional[str] = None


@dataclass
class DeadCodeFixResult:
    """Complete result of dead code fixing operation."""
    total_files_processed: int
    successful_files: int
    failed_files: int
    total_fixes_applied: int
    total_errors: int
    execution_time: float
    dry_run: bool
    timestamp: str
    file_results: List[FixResult]
    summary: Dict[str, Any]


class DeadCodeFixerPlugin(BasePlugin):
    """
    Plugin for automatically fixing dead code issues.
    
    This plugin focuses on high-confidence unused imports and provides
    conservative, safe fixes with comprehensive reporting.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "dead_code_fixer"
        self.description = "Automatically fix high-confidence dead code issues"
        self.category = "fixer"
        self.priority = "medium"
        self.version = "1.0.0"
        
        # Configuration
        self.min_confidence = 0.95  # Only fix very high confidence issues
        self.supported_issue_types = ["unused_import", "unused_import_from"]
        self.dry_run = False
        self.create_backups = True
        
        # State
        self.analyzer: Optional[ImprovedDeadCodeAnalyzer] = None
        self.issues: List[DeadCodeIssue] = []
        self.fix_results: List[FixResult] = []
        
        # Initialize logger
        import logging
        self.logger = logging.getLogger(__name__)
        
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin with user settings."""
        self.min_confidence = config.get("min_confidence", 0.95)
        self.dry_run = config.get("dry_run", False)
        self.create_backups = config.get("create_backups", True)
        self.supported_issue_types = config.get("supported_issue_types", ["unused_import", "unused_import_from"])
        
        self.logger.info(f"Configured with min_confidence={self.min_confidence}, dry_run={self.dry_run}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the dead code fixing operation.
        
        Args:
            context: Plugin execution context containing project information
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        
        try:
            # Extract context information
            project_root = Path(context.get("project_root", "."))
            report_path = context.get("dead_code_report_path")
            
            if not report_path:
                # Run analysis first if no report provided
                self.logger.info("No dead code report provided, running analysis first")
                self.analyzer = ImprovedDeadCodeAnalyzer()
                analysis_result = self.analyzer.analyze_directory(project_root)
                self.issues = analysis_result.issues
            else:
                # Load issues from existing report
                self.issues = self._load_issues_from_report(report_path)
            
            # Filter issues for fixing
            fixable_issues = self._filter_fixable_issues()
            self.logger.info(f"Found {len(fixable_issues)} fixable issues out of {len(self.issues)} total")
            
            if not fixable_issues:
                return self._create_empty_result(start_time)
            
            # Group issues by file
            issues_by_file = self._group_issues_by_file(fixable_issues)
            
            # Fix each file
            for file_path, file_issues in issues_by_file.items():
                result = self._fix_file(file_path, file_issues)
                self.fix_results.append(result)
            
            # Generate final result
            execution_time = time.time() - start_time
            final_result = self._generate_final_result(execution_time)
            
            self.logger.info(f"Dead code fixing completed: {final_result['summary']['total_fixes_applied']} fixes applied")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error during dead code fixing: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _load_issues_from_report(self, report_path: Union[str, Path]) -> List[DeadCodeIssue]:
        """Load issues from a dead code analysis report."""
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        issues = []
        for issue_data in report_data.get("issues", []):
            issue = DeadCodeIssue(
                file_path=issue_data["file_path"],
                line_number=issue_data["line_number"],
                issue_type=issue_data["issue_type"],
                name=issue_data["name"],
                description=issue_data["description"],
                confidence=issue_data["confidence"],
                severity=issue_data["severity"],
                is_public_api=issue_data.get("is_public_api", False),
                is_used_cross_file=issue_data.get("is_used_cross_file", False),
                is_abstract_interface=issue_data.get("is_abstract_interface", False),
                removal_impact=issue_data.get("removal_impact", "low"),
                dependencies=issue_data.get("dependencies", [])
            )
            issues.append(issue)
        
        return issues
    
    def _filter_fixable_issues(self) -> List[DeadCodeIssue]:
        """Filter issues that can be safely fixed."""
        fixable = []
        
        for issue in self.issues:
            # Only fix supported issue types
            if issue.issue_type not in self.supported_issue_types:
                continue
            
            # Only fix high confidence issues
            if issue.confidence < self.min_confidence:
                continue
            
            # Don't fix public API items
            if issue.is_public_api:
                continue
            
            # Don't fix items used cross-file
            if issue.is_used_cross_file:
                continue
            
            # Don't fix abstract/interface items
            if issue.is_abstract_interface:
                continue
            
            fixable.append(issue)
        
        return fixable
    
    def _group_issues_by_file(self, issues: List[DeadCodeIssue]) -> Dict[str, List[DeadCodeIssue]]:
        """Group issues by file path."""
        grouped = {}
        for issue in issues:
            if issue.file_path not in grouped:
                grouped[issue.file_path] = []
            grouped[issue.file_path].append(issue)
        return grouped
    
    def _fix_file(self, file_path: str, issues: List[DeadCodeIssue]) -> FixResult:
        """Fix dead code issues in a single file."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create backup if not dry run
            backup_path = None
            if not self.dry_run and self.create_backups:
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Identify lines to remove
            lines_to_remove = set()
            changes_made = []
            
            for issue in issues:
                if issue.issue_type in ["unused_import", "unused_import_from"]:
                    # Find the import line
                    import_line = self._find_import_line(tree, issue.name, issue.issue_type)
                    if import_line is not None:
                        lines_to_remove.add(import_line)
                        changes_made.append({
                            "type": "removed_import",
                            "name": issue.name,
                            "line": import_line,
                            "confidence": issue.confidence
                        })
            
            if not lines_to_remove:
                return FixResult(
                    file_path=file_path,
                    success=True,
                    issues_fixed=0,
                    changes_made=[],
                    backup_path=backup_path
                )
            
            # Remove lines
            lines = content.split('\n')
            new_lines = []
            
            for i, line in enumerate(lines):
                if i + 1 not in lines_to_remove:  # Line numbers are 1-based
                    new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            # Write back if not dry run
            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            
            return FixResult(
                file_path=file_path,
                success=True,
                issues_fixed=len(lines_to_remove),
                changes_made=changes_made,
                backup_path=backup_path
            )
            
        except Exception as e:
            return FixResult(
                file_path=file_path,
                success=False,
                issues_fixed=0,
                changes_made=[],
                error_message=str(e)
            )
    
    def _find_import_line(self, tree: ast.AST, name: str, issue_type: str) -> Optional[int]:
        """Find the line number of an import statement."""
        for node in ast.walk(tree):
            if issue_type == "unused_import" and isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == name:
                        return node.lineno
            elif issue_type == "unused_import_from" and isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == name:
                        return node.lineno
        return None
    
    def _create_empty_result(self, start_time: float) -> Dict[str, Any]:
        """Create result when no fixable issues are found."""
        return {
            "success": True,
            "total_files_processed": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_fixes_applied": 0,
            "total_errors": 0,
            "execution_time": time.time() - start_time,
            "dry_run": self.dry_run,
            "timestamp": datetime.now().isoformat(),
            "file_results": [],
            "summary": {
                "message": "No fixable issues found",
                "total_issues_analyzed": len(self.issues),
                "fixable_issues": 0
            }
        }
    
    def _generate_final_result(self, execution_time: float) -> Dict[str, Any]:
        """Generate the final result dictionary."""
        successful_files = sum(1 for result in self.fix_results if result.success)
        failed_files = len(self.fix_results) - successful_files
        total_fixes = sum(result.issues_fixed for result in self.fix_results)
        total_errors = sum(1 for result in self.fix_results if not result.success)
        
        summary = {
            "total_files_processed": len(self.fix_results),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_fixes_applied": total_fixes,
            "total_errors": total_errors,
            "success_rate": (successful_files / len(self.fix_results) * 100) if self.fix_results else 0,
            "average_fixes_per_file": total_fixes / len(self.fix_results) if self.fix_results else 0,
            "dry_run": self.dry_run,
            "min_confidence_used": self.min_confidence,
            "supported_issue_types": self.supported_issue_types
        }
        
        return {
            "success": True,
            "total_files_processed": len(self.fix_results),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_fixes_applied": total_fixes,
            "total_errors": total_errors,
            "execution_time": execution_time,
            "dry_run": self.dry_run,
            "timestamp": datetime.now().isoformat(),
            "file_results": [
                {
                    "file_path": result.file_path,
                    "success": result.success,
                    "issues_fixed": result.issues_fixed,
                    "changes_made": result.changes_made,
                    "error_message": result.error_message,
                    "backup_path": result.backup_path
                }
                for result in self.fix_results
            ],
            "summary": summary
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "version": self.version,
            "supported_issue_types": self.supported_issue_types,
            "min_confidence": self.min_confidence,
            "dry_run_supported": True,
            "backup_supported": True
        }
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors = []
        
        if "min_confidence" in config:
            if not isinstance(config["min_confidence"], (int, float)):
                errors.append("min_confidence must be a number")
            elif not 0 <= config["min_confidence"] <= 1:
                errors.append("min_confidence must be between 0 and 1")
        
        if "dry_run" in config:
            if not isinstance(config["dry_run"], bool):
                errors.append("dry_run must be a boolean")
        
        if "create_backups" in config:
            if not isinstance(config["create_backups"], bool):
                errors.append("create_backups must be a boolean")
        
        if "supported_issue_types" in config:
            if not isinstance(config["supported_issue_types"], list):
                errors.append("supported_issue_types must be a list")
            else:
                valid_types = ["unused_import", "unused_import_from", "unused_function", "unused_class"]
                for issue_type in config["supported_issue_types"]:
                    if issue_type not in valid_types:
                        errors.append(f"Invalid issue type: {issue_type}")
        
        return errors


def main():
    """Main entry point for standalone execution."""
    import argparse
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import List
    
    parser = argparse.ArgumentParser(description="Dead Code Auto-Fixer Plugin")
    parser.add_argument("--report", required=True, help="Path to dead code analysis report")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--min-confidence", type=float, default=0.95, help="Minimum confidence for fixes")
    parser.add_argument("--output", help="Output file for fix results")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and configure plugin
    plugin = DeadCodeFixerPlugin()
    config = {
        "dry_run": args.dry_run,
        "min_confidence": args.min_confidence
    }
    plugin.configure(config)
    
    # Execute plugin
    context = {
        "dead_code_report_path": args.report
    }
    
    result = plugin.execute(context)
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Print summary
    if result["success"]:
        summary = result["summary"]
        print(f"\nDead Code Fix Summary:")
        print(f"Files processed: {summary['total_files_processed']}")
        print(f"Successful: {summary['successful_files']}")
        print(f"Failed: {summary['failed_files']}")
        print(f"Fixes applied: {summary['total_fixes_applied']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print(f"Dry run: {result['dry_run']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
