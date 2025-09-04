"""
Plugin System for Code Quality Pipeline

This module provides a flexible plugin architecture for extending the
code quality pipeline with new tools and analyzers.
"""

from .base_plugin import (
    BasePlugin, PluginResult, PluginContext, PluginMetadata,
    PluginCategory, PluginPriority, FileProcessorPlugin, DirectoryProcessorPlugin
)
from .plugin_registry import PluginRegistry
from .plugin_manager import PluginManager
from .exceptions import (
    PluginError, PluginNotFoundError, PluginDependencyError,
    PluginExecutionError, PluginConfigurationError, PluginTimeoutError
)

__all__ = [
    'BasePlugin',
    'PluginResult', 
    'PluginContext',
    'PluginMetadata',
    'PluginCategory',
    'PluginPriority',
    'FileProcessorPlugin',
    'DirectoryProcessorPlugin',
    'PluginRegistry',
    'PluginManager',
    'PluginError',
    'PluginNotFoundError',
    'PluginDependencyError',
    'PluginExecutionError',
    'PluginConfigurationError',
    'PluginTimeoutError'
]