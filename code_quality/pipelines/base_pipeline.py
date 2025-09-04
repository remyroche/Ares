#!/usr/bin/env python3
"""
Base Pipeline Class - Enhanced with Plugin Architecture

This class provides common functionality to reduce redundancy across pipeline files
and integrates with the plugin system for extensible functionality.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import plugin system
from ..plugins import (
    PluginManager, PluginRegistry, PluginContext, PluginResult,
    PluginCategory, PluginPriority, BasePlugin
)
from ..utils.dependency_manager import dependency_manager


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    project_root: Path
    output_dir: Path
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_per_tool: int = 300
    retry_attempts: int = 3
    log_level: str = "INFO"
    dry_run: bool = False
    verbose: bool = False
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    plugin_categories: List[PluginCategory] = field(default_factory=list)
    plugin_priorities: List[PluginPriority] = field(default_factory=list)
    specific_plugins: List[str] = field(default_factory=list)
    exclude_plugins: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.project_root.exists():
            errors.append(f"Project root does not exist: {self.project_root}")
        
        if not self.project_root.is_dir():
            errors.append(f"Project root is not a directory: {self.project_root}")
        
        if self.max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        if self.timeout_per_tool < 1:
            errors.append("timeout_per_tool must be at least 1")
        
        if self.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of: {valid_log_levels}")
        
        return errors


class BasePipeline:
    """Enhanced base class for all pipeline implementations with plugin support."""
    
    def __init__(self, project_root: str = "/workspace/src", config: Optional[PipelineConfig] = None):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Initialize configuration
        if config is None:
            self.config = PipelineConfig(
                project_root=self.project_root,
                output_dir=self.reports_dir
            )
        else:
            self.config = config
        
        # Initialize plugin system
        self.plugin_registry = PluginRegistry()
        self.plugin_manager = PluginManager(self.plugin_registry)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize metrics
        self.metrics = {
            "execution_count": 0,
            "total_execution_time": 0.0,
            "successful_executions": 0,
            "failed_executions": 0,
            "plugins_used": set(),
            "files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0
        }
        
        # Discover and register plugins
        self._discover_plugins()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _discover_plugins(self) -> None:
        """Discover and register available plugins."""
        try:
            # Discover plugins in the examples directory
            examples_dir = Path(__file__).parent.parent / "plugins" / "examples"
            if examples_dir.exists():
                discovered = self.plugin_registry.discover_plugins(examples_dir)
                self.logger.info(f"Discovered {discovered} plugins")
            
            # Log plugin status
            available = self.plugin_registry.get_available_plugins()
            unavailable = self.plugin_registry.get_unavailable_plugins()
            
            self.logger.info(f"Available plugins: {available}")
            if unavailable:
                self.logger.warning(f"Unavailable plugins: {list(unavailable.keys())}")
                
        except Exception as e:
            self.logger.error(f"Failed to discover plugins: {e}")
    
    def register_plugin(self, plugin_class: type, instance: Optional[BasePlugin] = None) -> None:
        """Register a plugin with the pipeline."""
        try:
            self.plugin_registry.register_plugin(plugin_class, instance)
            self.logger.info(f"Registered plugin: {plugin_class.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugins."""
        return self.plugin_registry.get_available_plugins()
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get information about a specific plugin."""
        return self.plugin_registry.get_plugin_info(plugin_name)
    
    def execute_plugins(self, 
                       plugin_names: Optional[List[str]] = None,
                       categories: Optional[List[PluginCategory]] = None,
                       priorities: Optional[List[PluginPriority]] = None) -> Dict[str, Any]:
        """Execute plugins based on configuration."""
        # Find target files
        target_files = self._find_python_files()
        
        # Create plugin context
        context = PluginContext(
            project_root=self.project_root,
            target_files=target_files,
            configuration=self.config.__dict__,
            cache_dir=self.config.cache_dir,
            output_dir=self.config.output_dir,
            parallel_execution=self.config.parallel_execution,
            max_workers=self.config.max_workers,
            timeout=self.config.timeout_per_tool,
            dry_run=self.config.dry_run,
            verbose=self.config.verbose
        )
        
        # Execute plugins
        result = self.plugin_manager.execute_pipeline(
            plugin_names=plugin_names or self.config.specific_plugins,
            categories=categories or self.config.plugin_categories,
            priorities=priorities or self.config.plugin_priorities,
            context=context,
            parallel=self.config.parallel_execution,
            max_workers=self.config.max_workers,
            timeout_per_plugin=self.config.timeout_per_tool
        )
        
        # Update metrics
        self._update_metrics(result)
        
        return result
    
    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update pipeline metrics from plugin execution results."""
        self.metrics["execution_count"] += 1
        
        pipeline_info = result.get("pipeline_info", {})
        self.metrics["total_execution_time"] += pipeline_info.get("total_execution_time", 0)
        
        if pipeline_info.get("successful_plugins", 0) > 0:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update plugin usage
        for plugin_result in result.get("results", []):
            self.metrics["plugins_used"].add(plugin_result.get("plugin_name", ""))
        
        # Update file and issue counts
        summary = result.get("summary", {})
        self.metrics["files_processed"] += summary.get("total_files_processed", 0)
        self.metrics["issues_found"] += summary.get("total_issues_found", 0)
        self.metrics["issues_fixed"] += summary.get("total_issues_fixed", 0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics."""
        metrics = self.metrics.copy()
        metrics["plugins_used"] = list(metrics["plugins_used"])
        
        # Add plugin manager metrics
        plugin_metrics = self.plugin_manager.get_metrics()
        metrics["plugin_metrics"] = plugin_metrics
        
        return metrics
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.plugin_manager.get_execution_history(limit)
    
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
        """Generate a comprehensive summary of pipeline results."""
        return {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "pipeline_type": self.__class__.__name__,
            "configuration": {
                "parallel_execution": self.config.parallel_execution,
                "max_workers": self.config.max_workers,
                "timeout_per_tool": self.config.timeout_per_tool,
                "dry_run": self.config.dry_run,
                "log_level": self.config.log_level
            },
            "plugin_summary": {
                "available_plugins": self.get_available_plugins(),
                "plugin_categories": [cat.value for cat in self.config.plugin_categories],
                "plugin_priorities": [pri.value for pri in self.config.plugin_priorities],
                "specific_plugins": self.config.specific_plugins,
                "excluded_plugins": self.config.exclude_plugins
            },
            "results_summary": self._summarize_results(),
            "metrics": self.get_metrics()
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