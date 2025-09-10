#!/usr/bin/env python3
"""
Base Pipeline Class - Enhanced with Plugin Architecture and Standardized Initialization

This class provides common functionality to reduce redundancy across pipeline files
and integrates with the plugin system for extensible functionality. It also standardizes
common initialization patterns and naming conventions across all pipelines.
"""

import json
import time
import logging
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

# Import plugin system (with fallback)
try:
    from plugins import (
        PluginManager, PluginRegistry, PluginContext, PluginResult,
        PluginCategory, PluginPriority, BasePlugin
    )
    PLUGINS_AVAILABLE = True
except ImportError:
    # Create minimal fallback classes if plugins are not available
    class PluginManager:
        def __init__(self, *args, **kwargs):
            pass
    class PluginRegistry:
        def __init__(self, *args, **kwargs):
            pass
    class PluginContext:
        def __init__(self, *args, **kwargs):
            pass
    class PluginResult:
        def __init__(self, *args, **kwargs):
            pass
    class PluginCategory:
        pass
    class PluginPriority:
        pass
    class BasePlugin:
        pass
    PLUGINS_AVAILABLE = False


try:
    from pydantic import BaseModel, Field, validator, root_validator
    from typing import Optional, List
    from pathlib import Path

    class PipelineConfig(BaseModel):
        """Configuration for pipeline execution with Pydantic validation."""
        project_root: Path
        output_dir: Optional[Path] = None
        parallel_execution: bool = Field(default=True, description="Enable parallel execution")
        max_workers: int = Field(default=4, ge=1, le=32, description="Maximum number of worker threads")
        timeout_per_tool: int = Field(default=300, ge=1, le=3600, description="Timeout per tool in seconds")
        retry_attempts: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")
        log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level")
        dry_run: bool = Field(default=False, description="Run in dry-run mode")
        verbose: bool = Field(default=False, description="Enable verbose output")
        cache_enabled: bool = Field(default=True, description="Enable result caching")
        cache_dir: Optional[Path] = Field(default=None, description="Cache directory path")

        class Config:
            """Pydantic configuration."""
            validate_assignment = True
            arbitrary_types_allowed = True

        @validator('project_root')
        def validate_project_root(cls, v):
            """Validate project root exists and is a directory."""
            if not v.exists():
                raise ValueError(f"Project root does not exist: {v}")
            if not v.is_dir():
                raise ValueError(f"Project root is not a directory: {v}")
            return v

        @validator('output_dir', always=True)
        def set_output_dir(cls, v, values):
            """Set default output directory if not provided."""
            if v is None and 'project_root' in values:
                return values['project_root'] / "code_quality" / "reports"
            return v

        def dict(self, **kwargs):
            """Convert to dictionary with Path objects as strings."""
            data = super().dict(**kwargs)
            # Convert Path objects to strings for JSON serialization
            for key, value in data.items():
                if isinstance(value, Path):
                    data[key] = str(value)
            return data

except ImportError:
    # Fallback to dataclass if Pydantic is not available
    from dataclasses import dataclass, field

    @dataclass
    class PipelineConfig:
        """Configuration for pipeline execution (fallback without Pydantic)."""
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
        plugin_categories: List = field(default_factory=list)
        plugin_priorities: List = field(default_factory=list)
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
    """Enhanced base class for all pipeline implementations with plugin support and standardized initialization."""

    def __init__(self, project_root: Optional[Union[str, Path]] = None, config: Optional[PipelineConfig] = None,
                 enable_plugins: bool = True, pipeline_name: str = "base") -> None:
        # Initialize logging first (needed for error handling in other methods)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

        # Standardized initialization using helper methods
        self._setup_project_paths(project_root)
        self._setup_timestamps()
        self._setup_results()
        self._setup_reports_directory(pipeline_name)

        # Initialize configuration
        if config is None:
            self.config = PipelineConfig(
                project_root=self.project_root,
                output_dir=self.reports_dir
            )
        else:
            self.config = config

        # Initialize plugin system if enabled
        self.enable_plugins = enable_plugins
        if self.enable_plugins:
            self._setup_plugin_system()
        else:
            self.plugin_registry = None
            self.plugin_manager = None

        # Re-setup logging with proper configuration
        self.logger = self._setup_logging()

        # Initialize metrics
        self._setup_metrics()

        # Initialize caching
        self._setup_caching()

        # Discover and register plugins if enabled
        if self.enable_plugins:
            self._discover_plugins()

    def _setup_project_paths(self, project_root: Optional[Union[str, Path]]) -> None:
        """Standardized project path setup."""
        if project_root is None:
            self.project_root: Path = Path.cwd()
        else:
            self.project_root: Path = Path(project_root)

        # Ensure absolute path
        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()

    def _setup_timestamps(self) -> None:
        """Standardized timestamp setup."""
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def _setup_results(self) -> None:
        """Standardized results dictionary setup."""
        self.results: Dict[str, Any] = {}

    def _setup_reports_directory(self, pipeline_name: str) -> None:
        """Standardized reports directory setup."""
        try:
            self.reports_dir: Path = self.project_root / "code_quality" / "reports" / pipeline_name
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # Fallback to a temporary directory if the default location is not writable
            import tempfile
            temp_dir: Path = Path(tempfile.gettempdir()) / "code_quality_reports" / pipeline_name
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir = temp_dir
            self.logger.warning(f"Using fallback reports directory: {self.reports_dir} due to: {e}")

    def _setup_plugin_system(self) -> None:
        """Standardized plugin system setup."""
        if PLUGINS_AVAILABLE:
            self.plugin_registry: PluginRegistry = PluginRegistry()
            self.plugin_manager: PluginManager = PluginManager(self.plugin_registry)
        else:
            self.plugin_registry = None
            self.plugin_manager = None

    def _setup_metrics(self) -> None:
        """Standardized metrics setup."""
        self.metrics: Dict[str, Any] = {
            "execution_count": 0,
            "total_execution_time": 0.0,
            "successful_executions": 0,
            "failed_executions": 0,
            "plugins_used": set(),
            "files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0
        }

    def _setup_caching(self) -> None:
        """Initialize caching system."""
        from functools import lru_cache
        import hashlib

        self.cache_dir: Path = self.config.cache_dir or (self.project_root / ".cache" / "code_quality")
        self.cache_enabled: bool = self.config.cache_enabled
        self.cache: Dict[str, Any] = {}

        # Create cache directory if caching is enabled
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate a cache key for the given operation and arguments."""
        key_data = f"{operation}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and valid."""
        if not self.cache_enabled:
            return None

        # Check in-memory cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Check file-based cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Check if cache is still valid (optional: add timestamp validation)
                return cached_data.get('result')
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def _set_cached_result(self, cache_key: str, result: Any) -> None:
        """Cache the result."""
        if not self.cache_enabled:
            return

        # Store in memory cache
        self.cache[cache_key] = result

        # Store in file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            import json
            cache_data = {
                'result': result,
                'timestamp': self.timestamp,
                'pipeline': self.__class__.__name__
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception:
            # Silently fail if caching to disk fails
            pass

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger: logging.Logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Create formatter
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add console handler
        if not logger.handlers:
            console_handler: logging.StreamHandler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def _discover_plugins(self) -> None:
        """Discover and register available plugins."""
        try:
            # Discover plugins in the production directory first
            production_dir: Path = Path(__file__).parent.parent / "plugins" / "production"
            if production_dir.exists():
                discovered: int = self.plugin_registry.discover_plugins(production_dir)
                self.logger.info(f"Discovered {discovered} production plugins")

            # Also discover plugins in the examples directory as fallback
            examples_dir: Path = Path(__file__).parent.parent / "plugins" / "examples"
            if examples_dir.exists():
                discovered: int = self.plugin_registry.discover_plugins(examples_dir)
                self.logger.info(f"Discovered {discovered} example plugins")

            # Log plugin status
            available: List[str] = self.plugin_registry.get_available_plugins()
            unavailable: Dict[str, str] = self.plugin_registry.get_unavailable_plugins()

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

    def register_plugins_batch(self, plugin_classes: List[type]) -> None:
        """Register multiple plugins at once with standardized error handling."""
        if not self.enable_plugins or not self.plugin_registry:
            return

        registered_count: int = 0
        failed_plugins: List[tuple[str, str]] = []

        for plugin_class in plugin_classes:
            try:
                self.plugin_registry.register_plugin(plugin_class)
                registered_count += 1
                self.logger.debug(f"Registered plugin: {plugin_class.__name__}")
            except Exception as e:
                failed_plugins.append((plugin_class.__name__, str(e)))
                self.logger.warning(f"Failed to register plugin {plugin_class.__name__}: {e}")

        if registered_count > 0:
            self.logger.info(f"Successfully registered {registered_count} plugins")

        if failed_plugins:
            self.logger.warning(f"Failed to register {len(failed_plugins)} plugins: {[name for name, _ in failed_plugins]}")

    def setup_pipeline_paths(self) -> None:
        """Standardized pipeline path setup - call this in subclasses after initialization."""
        # Add parent directory to path (standardized pattern)
        parent_dir: Path = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        # Add current directory to path if needed
        current_dir: Path = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

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
        target_files: List[Path] = self._find_python_files()

        # Create plugin context
        context: PluginContext = PluginContext(
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
        result: Dict[str, Any] = self.plugin_manager.execute_pipeline(
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

        pipeline_info: Dict[str, Any] = result.get("pipeline_info", {})
        self.metrics["total_execution_time"] += pipeline_info.get("total_execution_time", 0)

        if pipeline_info.get("successful_plugins", 0) > 0:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1

        # Update plugin usage
        for plugin_result in result.get("results", []):
            self.metrics["plugins_used"].add(plugin_result.get("plugin_name", ""))

        # Update file and issue counts
        summary: Dict[str, Any] = result.get("summary", {})
        self.metrics["files_processed"] += summary.get("total_files_processed", 0)
        self.metrics["issues_found"] += summary.get("total_issues_found", 0)
        self.metrics["issues_fixed"] += summary.get("total_issues_fixed", 0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics."""
        metrics: Dict[str, Any] = self.metrics.copy()
        metrics["plugins_used"] = list(metrics["plugins_used"])

        # Add plugin manager metrics
        plugin_metrics: Dict[str, Any] = self.plugin_manager.get_metrics()
        metrics["plugin_metrics"] = plugin_metrics

        return metrics

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.plugin_manager.get_execution_history(limit)
    
    def _setup_execution_tracking(self) -> None:
        """Set up execution time tracking."""
        self.start_time = time.time()

    def _finalize_execution_tracking(self) -> float:
        """Finalize execution time tracking."""
        self.end_time = time.time()
        if self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def _save_report(self, data: Dict[str, Any], filename: str) -> Path:
        """Save a report to the reports directory."""
        report_path: Path = self.reports_dir / f"{filename}_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)
        return report_path

    def _print_section_header(self, title: str, width: int = 60) -> None:
        """Print a formatted section header."""
        print("\n" + "="*width)
        print(title)
        print("="*width)

    def _print_pipeline_header(self, pipeline_name: str, width: int = 80) -> None:
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
        summary: Dict[str, Any] = {
            "total_categories": len(self.results),
            "categories": list(self.results.keys())
        }

        # Count successful operations
        successful_ops: int = 0
        total_ops: int = 0

        for category, tools in self.results.items():
            if isinstance(tools, dict):
                for tool_name, result in tools.items():
                    total_ops += 1
                    if isinstance(result, dict):
                        if result.get("success", True):  # Default to True if not specified
                            successful_ops += 1

        summary["successful_operations"] = successful_ops
        summary["total_operations"] = total_ops
        summary["success_rate"] = (successful_ops / total_ops * 100) if total_ops > 0 else 0.0

        return summary

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print a formatted summary."""
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Pipeline: {summary['pipeline_type']}")
        print(f"Total execution time: {summary['total_execution_time']:.2f} seconds")

        results_summary: Dict[str, Any] = summary.get("results_summary", {})
        print(f"Categories processed: {results_summary.get('total_categories', 0)}")
        print(f"Operations: {results_summary.get('successful_operations', 0)}/{results_summary.get('total_operations', 0)} successful")
        print(f"Success rate: {results_summary.get('success_rate', 0):.1f}%")

        print(f"\nReports saved to: {self.reports_dir}")

    def _handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors consistently across pipelines."""
        error_info: Dict[str, Any] = {
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

    def _find_python_files(self, exclude_patterns: Optional[List[str]] = None, respect_gitignore: bool = True) -> List[Path]:
        """Find all Python files in the project root, respecting .gitignore patterns."""
        if exclude_patterns is None:
            exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env"]

        python_files: List[Path] = []
        try:
            for py_file in self.project_root.rglob("*.py"):
                # Check if file should be excluded by patterns
                should_exclude: bool = False
                for pattern in exclude_patterns:
                    if pattern in str(py_file):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

                # Check if file should be ignored by .gitignore
                if respect_gitignore:
                    from ..utils.gitignore_parser import should_ignore_file
                    if should_ignore_file(py_file, self.project_root):
                        continue

                python_files.append(py_file)

        except Exception as e:
            print(f"Error finding Python files: {e}")

        return python_files

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of a file before modification."""
        try:
            backup_path: Path = file_path.with_suffix(f"{file_path.suffix}.backup_{self.timestamp}")
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

    def health_check(self) -> Dict[str, Any]:
        """Return pipeline health status."""
        import psutil
        import os

        try:
            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0.0

        return {
            "status": "healthy" if self._validate_project_root() else "unhealthy",
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "project_root_valid": self._validate_project_root(),
            "reports_dir_exists": self.reports_dir.exists(),
            "reports_dir": str(self.reports_dir),
            "plugins_loaded": len(self.get_available_plugins()) if self.enable_plugins else 0,
            "memory_usage_mb": round(memory_mb, 2),
            "execution_count": self.metrics.get("execution_count", 0),
            "pipeline_type": self.__class__.__name__
        }

    def cleanup(self) -> None:
        """Cleanup resources used by the pipeline."""
        # Override in subclasses for specific cleanup needs
        pass

    def __enter__(self):
        """Context manager entry."""
        self._setup_execution_tracking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self._finalize_execution_tracking()
        self.cleanup()

        if exc_type:
            print(f"Pipeline exited with error: {exc_val}")
        else:
            print("Pipeline completed successfully")