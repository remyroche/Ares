"""
Base Plugin Classes

Core classes for the plugin system including base plugin interface,
plugin context, and plugin results.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime


class PluginCategory(Enum):
    """Categories for organizing plugins."""
    SYNTAX = "syntax"
    IMPORT = "import"
    LINTING = "linting"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    FORMATTING = "formatting"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


class PluginPriority(Enum):
    """Priority levels for plugin execution order."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    category: PluginCategory
    priority: PluginPriority = PluginPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    min_python_version: str = "3.8"
    max_python_version: Optional[str] = None
    required_packages: List[str] = field(default_factory=list)
    optional_packages: List[str] = field(default_factory=list)
    configuration_schema: Optional[Dict[str, Any]] = None


@dataclass
class PluginContext:
    """Context passed to plugins during execution."""
    project_root: Path
    target_files: List[Path]
    configuration: Dict[str, Any]
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    parallel_execution: bool = False
    max_workers: int = 4
    timeout: int = 300
    dry_run: bool = False
    verbose: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginResult:
    """Result returned by plugin execution."""
    plugin_name: str
    success: bool
    execution_time: float
    files_processed: int = 0
    files_fixed: int = 0
    files_failed: int = 0
    issues_found: int = 0
    issues_fixed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_metric(self, name: str, value: Any):
        """Add a metric."""
        self.metrics[name] = value
    
    def add_artifact(self, artifact_path: Path):
        """Add an output artifact."""
        self.artifacts.append(artifact_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "success": self.success,
            "execution_time": self.execution_time,
            "files_processed": self.files_processed,
            "files_fixed": self.files_fixed,
            "files_failed": self.files_failed,
            "issues_found": self.issues_found,
            "issues_fixed": self.issues_fixed,
            "errors": self.errors,
            "warnings": self.warnings,
            "output_data": self.output_data,
            "metrics": self.metrics,
            "artifacts": [str(artifact) for artifact in self.artifacts],
            "timestamp": self.timestamp.isoformat()
        }


class BasePlugin(ABC):
    """
    Base class for all pipeline plugins.
    
    Plugins should inherit from this class and implement the required methods.
    """
    
    def __init__(self, configuration: Optional[Dict[str, Any]] = None):
        self.configuration = configuration or {}
        self.metadata = self.get_metadata()
        self._validate_configuration()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.
        
        Returns:
            PluginMetadata: Plugin metadata including name, version, etc.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the plugin is available (dependencies met).
        
        Returns:
            bool: True if plugin can be executed, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, context: PluginContext) -> PluginResult:
        """
        Execute the plugin with the given context.
        
        Args:
            context: Plugin execution context
            
        Returns:
            PluginResult: Result of plugin execution
        """
        pass
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate plugin dependencies and return missing ones.
        
        Returns:
            List[str]: List of missing dependencies
        """
        missing = []
        
        # Check required packages
        for package in self.metadata.required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        return missing
    
    def get_supported_file_types(self) -> Set[str]:
        """
        Get file types supported by this plugin.
        
        Returns:
            Set[str]: Set of supported file extensions (e.g., {'.py', '.pyi'})
        """
        return {'.py'}  # Default to Python files
    
    def should_process_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be processed by this plugin.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file should be processed
        """
        return file_path.suffix in self.get_supported_file_types()
    
    def pre_execute(self, context: PluginContext) -> None:
        """
        Called before plugin execution.
        
        Args:
            context: Plugin execution context
        """
        pass
    
    def post_execute(self, context: PluginContext, result: PluginResult) -> None:
        """
        Called after plugin execution.
        
        Args:
            context: Plugin execution context
            result: Plugin execution result
        """
        pass
    
    def _validate_configuration(self) -> None:
        """Validate plugin configuration."""
        if self.metadata.configuration_schema:
            # Basic validation - could be enhanced with JSON schema validation
            for key, value in self.configuration.items():
                if key not in self.metadata.configuration_schema:
                    raise ValueError(f"Unknown configuration key: {key}")
    
    def _create_result(self, plugin_name: str, start_time: float) -> PluginResult:
        """Create a plugin result with timing information."""
        execution_time = time.time() - start_time
        return PluginResult(
            plugin_name=plugin_name,
            success=True,
            execution_time=execution_time
        )
    
    def _handle_error(self, error: Exception, result: PluginResult) -> None:
        """Handle errors during plugin execution."""
        result.add_error(f"{type(error).__name__}: {str(error)}")
        result.success = False
    
    def __str__(self) -> str:
        """String representation of the plugin."""
        return f"{self.metadata.name} v{self.metadata.version}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the plugin."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"category={self.metadata.category.value})"
        )


class FileProcessorPlugin(BasePlugin):
    """
    Base class for plugins that process individual files.
    
    Provides common functionality for file-based processing.
    """
    
    def execute(self, context: PluginContext) -> PluginResult:
        """Execute the plugin by processing each file."""
        start_time = time.time()
        result = self._create_result(self.metadata.name, start_time)
        
        try:
            self.pre_execute(context)
            
            # Filter files that should be processed
            files_to_process = [
                f for f in context.target_files
                if self.should_process_file(f)
            ]
            
            result.files_processed = len(files_to_process)
            
            for file_path in files_to_process:
                try:
                    file_result = self.process_file(file_path, context)
                    if file_result.get('success', False):
                        result.files_fixed += 1
                        result.issues_fixed += file_result.get('issues_fixed', 0)
                    else:
                        result.files_failed += 1
                        result.add_error(f"Failed to process {file_path}: {file_result.get('error', 'Unknown error')}")
                    
                    result.issues_found += file_result.get('issues_found', 0)
                    
                except Exception as e:
                    result.files_failed += 1
                    self._handle_error(e, result)
            
            self.post_execute(context, result)
            
        except Exception as e:
            self._handle_error(e, result)
        
        return result
    
    @abstractmethod
    def process_file(self, file_path: Path, context: PluginContext) -> Dict[str, Any]:
        """
        Process a single file.
        
        Args:
            file_path: Path to the file to process
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Result of processing the file
        """
        pass


class DirectoryProcessorPlugin(BasePlugin):
    """
    Base class for plugins that process entire directories.
    
    Provides common functionality for directory-based processing.
    """
    
    def execute(self, context: PluginContext) -> PluginResult:
        """Execute the plugin by processing the directory."""
        start_time = time.time()
        result = self._create_result(self.metadata.name, start_time)
        
        try:
            self.pre_execute(context)
            
            directory_result = self.process_directory(context.project_root, context)
            
            result.files_processed = directory_result.get('files_processed', 0)
            result.files_fixed = directory_result.get('files_fixed', 0)
            result.files_failed = directory_result.get('files_failed', 0)
            result.issues_found = directory_result.get('issues_found', 0)
            result.issues_fixed = directory_result.get('issues_fixed', 0)
            
            if directory_result.get('errors'):
                for error in directory_result['errors']:
                    result.add_error(error)
            
            if directory_result.get('warnings'):
                for warning in directory_result['warnings']:
                    result.add_warning(warning)
            
            result.output_data = directory_result.get('output_data', {})
            
            self.post_execute(context, result)
            
        except Exception as e:
            self._handle_error(e, result)
        
        return result
    
    @abstractmethod
    def process_directory(self, directory_path: Path, context: PluginContext) -> Dict[str, Any]:
        """
        Process a directory.
        
        Args:
            directory_path: Path to the directory to process
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Result of processing the directory
        """
        pass