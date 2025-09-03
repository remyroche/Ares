"""
Plugin system for code quality tools.
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CodeFixer(Protocol):
    """Protocol for code fixing tools."""

    def can_fix(self, file_path: str) -> bool:
        """Check if this tool can fix the given file."""
        ...

    def fix(self, file_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Fix issues in the given file."""
        ...

    def get_name(self) -> str:
        """Get the name of this tool."""
        ...

    def get_description(self) -> str:
        """Get a description of this tool."""
        ...


class CodeAnalyzer(Protocol):
    """Protocol for code analysis tools."""

    def can_analyze(self, file_path: str) -> bool:
        """Check if this tool can analyze the given file."""
        ...

    def analyze(self, file_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Analyze the given file."""
        ...

    def get_name(self) -> str:
        """Get the name of this tool."""
        ...

    def get_description(self) -> str:
        """Get a description of this tool."""
        ...


class BasePlugin(ABC):
    """Base class for all plugins."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this plugin."""

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this plugin."""

    @abstractmethod
    def get_version(self) -> str:
        """Get the version of this plugin."""

    def is_enabled(self) -> bool:
        """Check if this plugin is enabled."""
        return self.config.get("enabled", True)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value for this plugin."""
        return self.config.get(key, default)


class BaseCodeFixer(BasePlugin):
    """Base class for code fixing plugins."""

    @abstractmethod
    def can_fix(self, file_path: str) -> bool:
        """Check if this tool can fix the given file."""

    @abstractmethod
    def fix(self, file_path: str) -> dict[str, Any]:
        """Fix issues in the given file."""

    def get_supported_extensions(self) -> list[str]:
        """Get list of file extensions this fixer supports."""
        return [".py"]

    def get_fix_summary(self) -> dict[str, Any]:
        """Get a summary of what this fixer does."""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "version": self.get_version(),
            "supported_extensions": self.get_supported_extensions(),
        }


class BaseCodeAnalyzer(BasePlugin):
    """Base class for code analysis plugins."""

    @abstractmethod
    def can_analyze(self, file_path: str) -> bool:
        """Check if this tool can analyze the given file."""

    @abstractmethod
    def analyze(self, file_path: str) -> dict[str, Any]:
        """Analyze the given file."""

    def get_supported_extensions(self) -> list[str]:
        """Get list of file extensions this analyzer supports."""
        return [".py"]

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get a summary of what this analyzer does."""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "version": self.get_version(),
            "supported_extensions": self.get_supported_extensions(),
        }


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    description: str
    version: str
    plugin_type: str
    class_name: str
    module_path: str
    enabled: bool = True
    config: dict[str, Any] | None = None


class PluginManager:
    """Manages code quality tool plugins."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.plugins: dict[str, BaseCodeFixer | BaseCodeAnalyzer] = {}
        self.plugin_info: dict[str, PluginInfo] = {}
        self.logger = logging.getLogger(f"{__name__}.PluginManager")

        # Plugin directories to search
        self.plugin_dirs = [
            Path(__file__).parent.parent / "plugins",
            Path.cwd() / "plugins",
            Path.home() / ".code_quality" / "plugins",
        ]

    def discover_plugins(self) -> list[PluginInfo]:
        """Discover available plugins in plugin directories."""
        discovered_plugins = []

        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists() and plugin_dir.is_dir():
                discovered_plugins.extend(self._scan_directory(plugin_dir))

        return discovered_plugins

    def _scan_directory(self, directory: Path) -> list[PluginInfo]:
        """Scan a directory for plugins."""
        plugins = []

        for item in directory.iterdir():
            if item.is_file() and item.suffix == ".py":
                try:
                    plugin_info = self._load_plugin_from_file(item)
                    if plugin_info:
                        plugins.append(plugin_info)
                except Exception as e:
                    self.logger.warning(f"Failed to load plugin from {item}: {e}")
            elif item.is_dir() and (item / "__init__.py").exists():
                try:
                    plugin_info = self._load_plugin_from_package(item)
                    if plugin_info:
                        plugins.append(plugin_info)
                except Exception as e:
                    self.logger.warning(f"Failed to load plugin from package {item}: {e}")

        return plugins

    def _load_plugin_from_file(self, file_path: Path) -> PluginInfo | None:
        """Load a plugin from a Python file."""
        try:
            # Create module spec
            spec = importlib.util.spec_from_file_location(
                f"plugin_{file_path.stem}",
                file_path,
            )
            if spec is None or spec.loader is None:
                return None

            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, BaseCodeFixer | BaseCodeAnalyzer) and obj not in (BaseCodeFixer, BaseCodeAnalyzer)):

                    plugin_type = "fixer" if issubclass(obj, BaseCodeFixer) else "analyzer"

                    return PluginInfo(
                        name=obj().get_name(),
                        description=obj().get_description(),
                        version=obj().get_version(),
                        plugin_type=plugin_type,
                        class_name=name,
                        module_path=str(file_path),
                        enabled=True,
                    )

        except Exception as e:
            self.logger.debug(f"Failed to load plugin from {file_path}: {e}")

        return None

    def _load_plugin_from_package(self, package_path: Path) -> PluginInfo | None:
        """Load a plugin from a Python package."""
        try:
            # Import the package
            package_name = package_path.name
            spec = importlib.util.spec_from_file_location(
                package_name,
                package_path / "__init__.py",
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, BaseCodeFixer | BaseCodeAnalyzer) and obj not in (BaseCodeFixer, BaseCodeAnalyzer)):

                    plugin_type = "fixer" if issubclass(obj, BaseCodeFixer) else "analyzer"

                    return PluginInfo(
                        name=obj().get_name(),
                        description=obj().get_description(),
                        version=obj().get_version(),
                        plugin_type=plugin_type,
                        class_name=name,
                        module_path=str(package_path),
                        enabled=True,
                    )

        except Exception as e:
            self.logger.debug(f"Failed to load plugin from package {package_path}: {e}")

        return None

    def register_plugin(self, name: str, plugin: BaseCodeFixer | BaseCodeAnalyzer) -> None:
        """Register a plugin manually."""
        self.plugins[name] = plugin
        self.logger.info(f"Registered plugin: {name}")

    def get_plugin(self, name: str) -> BaseCodeFixer | BaseCodeAnalyzer | None:
        """Get a plugin by name."""
        return self.plugins.get(name)

    def get_all_plugins(self) -> dict[str, BaseCodeFixer | BaseCodeAnalyzer]:
        """Get all registered plugins."""
        return self.plugins.copy()

    def get_fixers(self) -> list[BaseCodeFixer]:
        """Get all registered code fixers."""
        return [plugin for plugin in self.plugins.values()
                if isinstance(plugin, BaseCodeFixer)]

    def get_analyzers(self) -> list[BaseCodeAnalyzer]:
        """Get all registered code analyzers."""
        return [plugin for plugin in self.plugins.values()
                if isinstance(plugin, BaseCodeAnalyzer)]

    def get_available_fixers(self, file_path: str) -> list[BaseCodeFixer]:
        """Get all fixers that can fix the given file."""
        return [fixer for fixer in self.get_fixers()
                if fixer.is_enabled() and fixer.can_fix(file_path)]

    def get_available_analyzers(self, file_path: str) -> list[BaseCodeAnalyzer]:
        """Get all analyzers that can analyze the given file."""
        return [analyzer for analyzer in self.get_analyzers()
                if analyzer.is_enabled() and analyzer.can_analyze(file_path)]

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self.plugins:
            self.plugins[name].config["enabled"] = True
            self.logger.info(f"Enabled plugin: {name}")
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        if name in self.plugins:
            self.plugins[name].config["enabled"] = False
            self.logger.info(f"Disabled plugin: {name}")
            return True
        return False

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all plugins with their information."""
        plugin_list = []

        for name, plugin in self.plugins.items():
            if isinstance(plugin, BaseCodeFixer):
                plugin_type = "fixer"
                summary = plugin.get_fix_summary()
            else:
                plugin_type = "analyzer"
                summary = plugin.get_analysis_summary()

            plugin_list.append({
                "name": name,
                "type": plugin_type,
                "enabled": plugin.is_enabled(),
                "description": summary.get("description", ""),
                "version": summary.get("version", ""),
                "supported_extensions": summary.get("supported_extensions", []),
            })

        return plugin_list
