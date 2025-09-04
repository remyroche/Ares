#!/usr/bin/env python3
"""
Base Pipeline Class - Common functionality for all pipeline implementations.

This class provides common functionality to reduce redundancy across pipeline files.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class BasePipeline:
    """Base class for all pipeline implementations."""
    
    def __init__(self, project_root: str = "/workspace/src"):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def _setup_execution_tracking(self):
        """Set up execution time tracking."""
        self.start_time = time.time()
    
    def _finalize_execution_tracking(self):
        """Finalize execution time tracking."""
        self.end_time = time.time()
        if self.start_time:
            return self.end_time - self.start_time
        return 0
    
    def _save_report(self, data: Dict[str, Any], filename: str) -> Path:
        """Save a report to the reports directory."""
        report_path = self.reports_dir / f"{filename}_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)
        return report_path
    
    def _print_section_header(self, title: str, width: int = 60):
        """Print a formatted section header."""
        print("\n" + "="*width)
        print(title)
        print("="*width)
    
    def _print_pipeline_header(self, pipeline_name: str, width: int = 80):
        """Print a formatted pipeline header."""
        print(f"\n{'='*width}")
        print(f"{pipeline_name.upper()}")
        print(f"{'='*width}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate a basic summary of pipeline results."""
        return {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "pipeline_type": self.__class__.__name__,
            "results_summary": self._summarize_results()
        }
    
    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize the results dictionary."""
        summary = {
            "total_categories": len(self.results),
            "categories": list(self.results.keys())
        }
        
        # Count successful operations
        successful_ops = 0
        total_ops = 0
        
        for category, tools in self.results.items():
            if isinstance(tools, dict):
                for tool_name, result in tools.items():
                    total_ops += 1
                    if isinstance(result, dict):
                        if result.get("success", True):  # Default to True if not specified
                            successful_ops += 1
        
        summary["successful_operations"] = successful_ops
        summary["total_operations"] = total_ops
        summary["success_rate"] = (successful_ops / total_ops * 100) if total_ops > 0 else 0
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary."""
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Pipeline: {summary['pipeline_type']}")
        print(f"Total execution time: {summary['total_execution_time']:.2f} seconds")
        
        results_summary = summary.get("results_summary", {})
        print(f"Categories processed: {results_summary.get('total_categories', 0)}")
        print(f"Operations: {results_summary.get('successful_operations', 0)}/{results_summary.get('total_operations', 0)} successful")
        print(f"Success rate: {results_summary.get('success_rate', 0):.1f}%")
        
        print(f"\nReports saved to: {self.reports_dir}")
    
    def _handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors consistently across pipelines."""
        error_info = {
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Error in {context}: {error}")
        return error_info
    
    def _validate_project_root(self) -> bool:
        """Validate that the project root exists and is accessible."""
        try:
            if not self.project_root.exists():
                print(f"Warning: Project root does not exist: {self.project_root}")
                return False
            
            if not self.project_root.is_dir():
                print(f"Warning: Project root is not a directory: {self.project_root}")
                return False
            
            # Try to list contents to check accessibility
            list(self.project_root.iterdir())
            return True
            
        except Exception as e:
            print(f"Error accessing project root {self.project_root}: {e}")
            return False
    
    def _find_python_files(self, exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Find all Python files in the project root."""
        if exclude_patterns is None:
            exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env"]
        
        python_files = []
        try:
            for py_file in self.project_root.rglob("*.py"):
                # Check if file should be excluded
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in str(py_file):
                        should_exclude = True
                        break
                
                if not should_exclude:
                    python_files.append(py_file)
                    
        except Exception as e:
            print(f"Error finding Python files: {e}")
        
        return python_files
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of a file before modification."""
        try:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{self.timestamp}")
            backup_path.write_text(file_path.read_text())
            return backup_path
        except Exception as e:
            print(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def _restore_backup(self, file_path: Path, backup_path: Path) -> bool:
        """Restore a file from backup."""
        try:
            file_path.write_text(backup_path.read_text())
            backup_path.unlink()  # Remove backup after successful restore
            return True
        except Exception as e:
            print(f"Failed to restore backup for {file_path}: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources used by the pipeline."""
        # Override in subclasses for specific cleanup needs
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self._setup_execution_tracking()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._finalize_execution_tracking()
        self.cleanup()
        
        if exc_type:
            print(f"Pipeline exited with error: {exc_val}")
        else:
            print("Pipeline completed successfully")