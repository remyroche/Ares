"""
Plugin System Exceptions

Custom exceptions for the plugin system to provide clear error handling.
"""


class PluginError(Exception):
    """Base exception for all plugin-related errors."""
    
    def __init__(self, message: str, plugin_name: str = None, error_code: str = None):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.error_code = error_code


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found."""
    
    def __init__(self, plugin_name: str):
        super().__init__(
            f"Plugin '{plugin_name}' not found in registry",
            plugin_name=plugin_name,
            error_code="PLUGIN_NOT_FOUND"
        )


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies are not met."""
    
    def __init__(self, plugin_name: str, missing_dependencies: list):
        super().__init__(
            f"Plugin '{plugin_name}' missing dependencies: {', '.join(missing_dependencies)}",
            plugin_name=plugin_name,
            error_code="PLUGIN_DEPENDENCY_ERROR"
        )
        self.missing_dependencies = missing_dependencies


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    
    def __init__(self, plugin_name: str, original_error: Exception):
        super().__init__(
            f"Plugin '{plugin_name}' execution failed: {str(original_error)}",
            plugin_name=plugin_name,
            error_code="PLUGIN_EXECUTION_ERROR"
        )
        self.original_error = original_error


class PluginConfigurationError(PluginError):
    """Raised when plugin configuration is invalid."""
    
    def __init__(self, plugin_name: str, config_error: str):
        super().__init__(
            f"Plugin '{plugin_name}' configuration error: {config_error}",
            plugin_name=plugin_name,
            error_code="PLUGIN_CONFIG_ERROR"
        )


class PluginTimeoutError(PluginError):
    """Raised when plugin execution times out."""
    
    def __init__(self, plugin_name: str, timeout_seconds: int):
        super().__init__(
            f"Plugin '{plugin_name}' timed out after {timeout_seconds} seconds",
            plugin_name=plugin_name,
            error_code="PLUGIN_TIMEOUT"
        )