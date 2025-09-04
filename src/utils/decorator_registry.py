from __future__ import annotations
'Central registry for all decorators with metadata and versioning and comprehensive error handling.'
import logging
from datetime import datetime
from typing import Callable, Any, Dict, List, Optional

# Import enhanced logging functions
try:
    from .logger import log_error_with_context, log_system_status, log_validation_result
    from .warning_symbols import error, warning, info, success
except ImportError:
    # Fallback if imports fail
    def log_error_with_context(logger, error, context=None, operation="", recovery_attempted=False):
        logger.error(f"Error in {operation}: {error}")
    
    def log_system_status(logger, component, status, details="", health_metrics=None):
        logger.info(f"System Status | {component} | {status}")
    
    def log_validation_result(logger, validator_name, result, details="", metrics=None):
        status = "PASSED" if result else "FAILED"
        logger.info(f"Validation {status} | {validator_name}")
    
    def error(msg): return f"❌ {msg}"
    def warning(msg): return f"⚠️ {msg}"
    def info(msg): return f"ℹ️ {msg}"
    def success(msg): return f"✅ {msg}"

logger = logging.getLogger(__name__)

class DecoratorMetadata:
    """Metadata for a registered decorator with comprehensive error handling."""

    def __init__(self, name: str, decorator: Callable, version: str='1.0', description: str='', tags: list[str]=None, deprecated: bool=False) -> None:
        try:
            logger.info(f"🔧 Creating DecoratorMetadata for '{name}' v{version}")
            
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Decorator name must be a non-empty string")
            
            if not callable(decorator):
                raise ValueError("Decorator must be callable")
            
            if not version or not isinstance(version, str):
                raise ValueError("Version must be a non-empty string")
            
            self.name = name
            self.decorator = decorator
            self.version = version
            self.description = description or ""
            self.tags = tags or []
            self.deprecated = bool(deprecated)
            self.registered_at = datetime.now()
            self.usage_count = 0
            self.last_used = None
            self.error_count = 0
            self.last_error = None
            
            logger.success(f"✅ DecoratorMetadata created successfully for '{name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to create DecoratorMetadata for '{name}': {e}")
            log_error_with_context(
                logger, e,
                context={"name": name, "version": version, "deprecated": deprecated},
                operation="DecoratorMetadata.__init__"
            )
            raise

    def __repr__(self) -> str:
        """String representation with comprehensive information."""
        try:
            status = "DEPRECATED" if self.deprecated else "ACTIVE"
            return f"DecoratorMetadata(name='{self.name}', version='{self.version}', status={status}, usage={self.usage_count}, errors={self.error_count})"
        except Exception as e:
            logger.error(f"❌ Error in DecoratorMetadata.__repr__: {e}")
            return f"DecoratorMetadata(name='{getattr(self, 'name', 'UNKNOWN')}', error='{e}')"

    def increment_usage(self) -> None:
        """Increment usage count and update last used timestamp."""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()
            logger.debug(f"📊 Usage incremented for decorator '{self.name}' (total: {self.usage_count})")
        except Exception as e:
            logger.error(f"❌ Error incrementing usage for decorator '{self.name}': {e}")

    def record_error(self, error: Exception) -> None:
        """Record an error that occurred with this decorator."""
        try:
            self.error_count += 1
            self.last_error = {
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.now()
            }
            logger.warning(f"⚠️ Error recorded for decorator '{self.name}' (total errors: {self.error_count})")
        except Exception as e:
            logger.error(f"❌ Error recording error for decorator '{self.name}': {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of this decorator."""
        try:
            logger.info(f"🏥 Getting health status for decorator '{self.name}'")
            
            # Calculate health score
            health_score = 100
            issues = []
            
            if self.deprecated:
                health_score -= 20
                issues.append("Decorator is deprecated")
            
            if self.error_count > 0:
                health_score -= min(self.error_count * 10, 50)
                issues.append(f"Has {self.error_count} recorded errors")
            
            if self.usage_count == 0:
                health_score -= 10
                issues.append("Never used")
            
            # Determine status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 70:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            else:
                status = "poor"
            
            health_info = {
                "name": self.name,
                "version": self.version,
                "status": status,
                "health_score": health_score,
                "usage_count": self.usage_count,
                "error_count": self.error_count,
                "deprecated": self.deprecated,
                "issues": issues,
                "last_used": self.last_used.isoformat() if self.last_used else None,
                "last_error": self.last_error
            }
            
            log_system_status(
                logger, f"Decorator-{self.name}", status,
                details=f"Health Score: {health_score}/100",
                health_metrics=health_info
            )
            
            return health_info
            
        except Exception as e:
            logger.error(f"❌ Error getting health status for decorator '{self.name}': {e}")
            log_error_with_context(
                logger, e,
                context={"name": self.name},
                operation="DecoratorMetadata.get_health_status"
            )
            return {
                "name": self.name,
                "status": "error",
                "health_score": 0,
                "error": str(e)
            }

class DecoratorRegistry:
    """Central registry for all decorators with metadata and versioning and comprehensive error handling."""

    def __init__(self) -> None:
        try:
            logger.info("🔧 Initializing DecoratorRegistry")
            self._decorators: dict[str, DecoratorMetadata] = {}
            self._aliases: dict[str, str] = {}
            self._version_history: dict[str, list[str]] = {}
            self._registry_stats = {
                "total_registrations": 0,
                "total_errors": 0,
                "last_updated": datetime.now()
            }
            logger.info("✅ DecoratorRegistry initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DecoratorRegistry: {e}")
            log_error_with_context(
                logger, e,
                operation="DecoratorRegistry.__init__"
            )
            raise

    def register(self, name: str, decorator: Callable, version: str='1.0', description: str='', tags: list[str]=None, deprecated: bool=False, aliases: list[str]=None) -> None:
        """Register a decorator with version tracking and comprehensive error handling."""
        try:
            logger.info(f"🔧 Registering decorator '{name}' v{version}")
            
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Decorator name must be a non-empty string")
            
            if not callable(decorator):
                raise ValueError("Decorator must be callable")
            
            if not version or not isinstance(version, str):
                raise ValueError("Version must be a non-empty string")
            
            # Check for existing decorator
            if name in self._decorators:
                existing = self._decorators[name]
                if existing.version != version:
                    if name not in self._version_history:
                        self._version_history[name] = []
                    self._version_history[name].append(existing.version)
                    logger.info(f'🔄 Updated decorator {name} from version {existing.version} to {version}')
                else:
                    logger.warning(f'⚠️ Decorator {name} v{version} already registered, updating metadata')
            
            # Create metadata with error handling
            try:
                metadata = DecoratorMetadata(name, decorator, version, description, tags, deprecated)
            except Exception as e:
                logger.error(f"❌ Failed to create metadata for decorator '{name}': {e}")
                raise
            
            # Register the decorator
            self._decorators[name] = metadata
            
            # Handle aliases
            if aliases:
                for alias in aliases:
                    if not alias or not isinstance(alias, str):
                        logger.warning(f"⚠️ Invalid alias '{alias}' for decorator '{name}', skipping")
                        continue
                    if alias in self._aliases and self._aliases[alias] != name:
                        logger.warning(f"⚠️ Alias '{alias}' already exists for decorator '{self._aliases[alias]}', overwriting for '{name}'")
                    self._aliases[alias] = name
                    logger.debug(f"🔗 Registered alias '{alias}' for decorator '{name}'")
            
            # Update registry stats
            self._registry_stats["total_registrations"] += 1
            self._registry_stats["last_updated"] = datetime.now()
            
            logger.info(f'✅ Registered decorator: {name} v{version}')
            
        except Exception as e:
            self._registry_stats["total_errors"] += 1
            logger.error(f"❌ Failed to register decorator '{name}': {e}")
            log_error_with_context(
                logger, e,
                context={
                    "name": name,
                    "version": version,
                    "deprecated": deprecated,
                    "aliases": aliases
                },
                operation="DecoratorRegistry.register"
            )
            raise

    def get(self, name: str, version: str=None) -> Callable:
        """Get decorator by name and optional version with comprehensive error handling."""
        try:
            logger.info(f"🔍 Getting decorator '{name}'" + (f" v{version}" if version else ""))
            
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Decorator name must be a non-empty string")
            
            # Resolve alias
            original_name = name
            if name in self._aliases:
                name = self._aliases[name]
                logger.debug(f"🔗 Resolved alias '{original_name}' to '{name}'")
            
            # Check if decorator exists
            if name not in self._decorators:
                available_decorators = list(self._decorators.keys())
                msg = f"Decorator '{name}' not found in registry. Available decorators: {available_decorators}"
                logger.error(f"❌ {msg}")
                raise KeyError(msg)
            
            metadata = self._decorators[name]
            
            # Check version if specified
            if version and metadata.version != version:
                msg = f'Version mismatch for {name}: requested {version}, available {metadata.version}'
                logger.error(f"❌ {msg}")
                raise ValueError(msg)
            
            # Increment usage and return decorator
            metadata.increment_usage()
            logger.info(f"✅ Retrieved decorator '{name}' v{metadata.version}")
            return metadata.decorator
            
        except Exception as e:
            self._registry_stats["total_errors"] += 1
            logger.error(f"❌ Failed to get decorator '{name}': {e}")
            log_error_with_context(
                logger, e,
                context={"name": name, "version": version},
                operation="DecoratorRegistry.get"
            )
            raise

    def list_decorators(self, include_deprecated: bool=False, tags: list[str]=None) -> list[DecoratorMetadata]:
        """List all registered decorators with optional filtering."""
        decorators = list(self._decorators.values())
        if not include_deprecated:
            decorators = [d for d in decorators if not d.deprecated]
        if tags:
            decorators = [d for d in decorators if any((tag in d.tags for tag in tags))]
        return sorted(decorators, key=lambda x: x.name)

    def get_usage_stats(self) -> dict[str, int]:
        """Get usage statistics for all decorators."""
        return {name: metadata.usage_count for name, metadata in self._decorators.items()}

    def deprecate(self, name: str, replacement: str=None) -> None:
        """Mark a decorator as deprecated."""
        if name in self._decorators:
            self._decorators[name].deprecated = True
            if replacement:
                logger.warning(f"Decorator '{name}' is deprecated. Use '{replacement}' instead.")
        else:
            msg = f"Decorator '{name}' not found in registry"
            raise KeyError(msg)

    def remove(self, name: str) -> None:
        """Remove a decorator from the registry."""
        if name in self._decorators:
            del self._decorators[name]
            aliases_to_remove = [alias for alias, target in self._aliases.items() if target == name]
            for alias in aliases_to_remove:
                del self._aliases[alias]
            logger.info(f'Removed decorator: {name}')
        else:
            msg = f"Decorator '{name}' not found in registry"
            raise KeyError(msg)

    def search(self, query: str) -> list[DecoratorMetadata]:
        """Search decorators by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for metadata in self._decorators.values():
            if query_lower in metadata.name.lower() or query_lower in metadata.description.lower() or any((query_lower in tag.lower() for tag in metadata.tags)):
                results.append(metadata)
        return results

    def export_config(self) -> dict[str, Any]:
        """Export registry configuration for persistence."""
        return {'decorators': {name: {'version': meta.version, 'description': meta.description, 'tags': meta.tags, 'deprecated': meta.deprecated, 'registered_at': meta.registered_at.isoformat(), 'usage_count': meta.usage_count} for name, meta in self._decorators.items()}, 'aliases': self._aliases, 'version_history': self._version_history}

    def get_registry_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the entire registry."""
        try:
            logger.info("🏥 Getting DecoratorRegistry health status")
            
            total_decorators = len(self._decorators)
            deprecated_count = sum(1 for d in self._decorators.values() if d.deprecated)
            total_errors = sum(d.error_count for d in self._decorators.values())
            total_usage = sum(d.usage_count for d in self._decorators.values())
            
            # Calculate health score
            health_score = 100
            issues = []
            
            if total_errors > 0:
                health_score -= min(total_errors * 5, 30)
                issues.append(f"Registry has {total_errors} total errors")
            
            if deprecated_count > total_decorators * 0.5:
                health_score -= 20
                issues.append(f"High percentage of deprecated decorators ({deprecated_count}/{total_decorators})")
            
            if total_usage == 0:
                health_score -= 15
                issues.append("No decorators have been used")
            
            # Determine overall status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 70:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            else:
                status = "poor"
            
            health_info = {
                "status": status,
                "health_score": health_score,
                "total_decorators": total_decorators,
                "deprecated_count": deprecated_count,
                "total_errors": total_errors,
                "total_usage": total_usage,
                "total_aliases": len(self._aliases),
                "registry_stats": self._registry_stats,
                "issues": issues,
                "decorator_health": [d.get_health_status() for d in self._decorators.values()]
            }
            
            # Only log health status if there are issues (fair/poor)
            if status in ["fair", "poor"]:
                log_system_status(
                    logger, "DecoratorRegistry", status,
                    details=f"Health Score: {health_score}/100 | Decorators: {total_decorators} | Errors: {total_errors}",
                    health_metrics=health_info
                )
            
            return health_info
            
        except Exception as e:
            logger.error(f"❌ Error getting registry health status: {e}")
            log_error_with_context(
                logger, e,
                operation="DecoratorRegistry.get_registry_health_status"
            )
            return {
                "status": "error",
                "health_score": 0,
                "error": str(e)
            }

    def validate_registry(self) -> tuple[bool, List[str]]:
        """Validate the entire registry and return validation results."""
        try:
            logger.info("🔍 Validating DecoratorRegistry")
            
            issues = []
            
            # Check for duplicate aliases
            alias_values = list(self._aliases.values())
            if len(alias_values) != len(set(alias_values)):
                issues.append("Duplicate aliases found")
            
            # Check for decorators with no usage
            unused_decorators = [name for name, meta in self._decorators.items() if meta.usage_count == 0]
            if unused_decorators:
                issues.append(f"Unused decorators: {unused_decorators}")
            
            # Check for decorators with high error rates
            high_error_decorators = [name for name, meta in self._decorators.items() if meta.error_count > 5]
            if high_error_decorators:
                issues.append(f"Decorators with high error rates: {high_error_decorators}")
            
            # Check for orphaned aliases
            orphaned_aliases = [alias for alias, target in self._aliases.items() if target not in self._decorators]
            if orphaned_aliases:
                issues.append(f"Orphaned aliases: {orphaned_aliases}")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.success("✅ DecoratorRegistry validation passed")
            else:
                logger.warning(f"⚠️ DecoratorRegistry validation failed with {len(issues)} issues")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"❌ Error during registry validation: {e}")
            log_error_with_context(
                logger, e,
                operation="DecoratorRegistry.validate_registry"
            )
            return False, [f"Validation error: {e}"]


def register_decorator(name: str, version: str='1.0', description: str='', tags: list[str]=None, deprecated: bool=False, aliases: list[str]=None) -> Callable:
    """Decorator to register a decorator function in the registry with comprehensive error handling."""

    def decorator(func: Callable) -> Callable:
        try:
            logger.info(f"🔧 Registering decorator function '{name}' using @register_decorator")
            decorator_registry.register(
                name=name, 
                decorator=func, 
                version=version, 
                description=description, 
                tags=tags or [], 
                deprecated=deprecated, 
                aliases=aliases or []
            )
            logger.success(f"✅ Decorator function '{name}' registered successfully")
            return func
        except Exception as e:
            logger.error(f"❌ Failed to register decorator function '{name}': {e}")
            log_error_with_context(
                logger, e,
                context={"name": name, "version": version, "function": func.__name__},
                operation="register_decorator"
            )
            # Return the function anyway to avoid breaking the decorator chain
            return func
    return decorator


# Global registry instance
decorator_registry = DecoratorRegistry()