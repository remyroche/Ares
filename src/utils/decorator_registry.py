"""Central registry for all decorators with metadata and versioning."""

from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)

class DecoratorCategory(Enum):
    """Categories for decorators."""
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    LOGGING = "logging"
    SECURITY = "security"
    CACHING = "caching"
    ERROR_HANDLING = "error_handling"
    MONITORING = "monitoring"
    UTILITY = "utility"
    LEGACY = "legacy"

@dataclass
class DecoratorMetadata:
    """Metadata for a registered decorator."""
    
    name: str
    decorator: Callable
    version: str
    description: str
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    registered_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    category: DecoratorCategory = DecoratorCategory.UTILITY
    migration_target: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    performance_impact: str = "low"  # low, medium, high
    security_level: str = "safe"  # safe, warning, dangerous
    
    def __repr__(self) -> str:
        """String representation of metadata."""
        return f"DecoratorMetadata(name='{self.name}', version='{self.version}', deprecated={self.deprecated})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "deprecated": self.deprecated,
            "registered_at": self.registered_at.isoformat(),
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "category": self.category.value,
            "migration_target": self.migration_target,
            "dependencies": self.dependencies,
            "performance_impact": self.performance_impact,
            "security_level": self.security_level
        }
    
    def update_usage(self) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()

class DecoratorRegistry:
    """Central registry for all decorators with metadata and versioning."""
    
    def __init__(self):
        """Initialize the decorator registry."""
        self._decorators: Dict[str, DecoratorMetadata] = {}
        self._aliases: Dict[str, str] = {}
        self._version_history: Dict[str, List[str]] = {}
        self._categories: Dict[DecoratorCategory, List[str]] = {cat: [] for cat in DecoratorCategory}
        self.logger = logging.getLogger("DecoratorRegistry")
        
        # Registry statistics
        self._total_registrations = 0
        self._total_usage = 0
        self._last_cleanup = datetime.now()
    
    def register(
        self, 
        name: str, 
        decorator: Callable, 
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        deprecated: bool = False,
        aliases: Optional[List[str]] = None,
        category: DecoratorCategory = DecoratorCategory.UTILITY,
        migration_target: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        performance_impact: str = "low",
        security_level: str = "safe"
    ) -> None:
        """Register a decorator function."""
        try:
            # Validate inputs
            if not name or not callable(decorator):
                raise ValueError("Name must be a string and decorator must be callable")
            
            # Check if decorator already exists
            if name in self._decorators:
                # Update existing decorator
                existing = self._decorators[name]
                if existing.version != version:
                    if name not in self._version_history:
                        self._version_history[name] = []
                    self._version_history[name].append(existing.version)
                    self.logger.info(f"Updated decorator {name} from version {existing.version} to {version}")
            
            # Create metadata
            metadata = DecoratorMetadata(
                name=name,
                decorator=decorator,
                version=version,
                description=description,
                tags=tags or [],
                deprecated=deprecated,
                category=category,
                migration_target=migration_target,
                dependencies=dependencies or [],
                performance_impact=performance_impact,
                security_level=security_level
            )
            
            # Register decorator
            self._decorators[name] = metadata
            self._categories[category].append(name)
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
            
            self._total_registrations += 1
            self.logger.info(f"Decorator {name} registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register decorator {name}: {e}")
            raise
    
    def get(self, name: str) -> Optional[Callable]:
        """Get a decorator by name."""
        # Check direct name
        if name in self._decorators:
            metadata = self._decorators[name]
            metadata.update_usage()
            self._total_usage += 1
            return metadata.decorator
        
        # Check aliases
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self.get(actual_name)
        
        return None
    
    def get_metadata(self, name: str) -> Optional[DecoratorMetadata]:
        """Get metadata for a decorator."""
        if name in self._decorators:
            return self._decorators[name]
        elif name in self._aliases:
            actual_name = self._aliases[name]
            return self._decorators.get(actual_name)
        return None
    
    def list_decorators(
        self, 
        include_deprecated: bool = True,
        category: Optional[DecoratorCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """List all registered decorators with optional filtering."""
        decorators = []
        
        for name, metadata in self._decorators.items():
            # Skip deprecated if not requested
            if not include_deprecated and metadata.deprecated:
                continue
            
            # Filter by category
            if category and metadata.category != category:
                continue
            
            # Filter by tags
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            decorators.append(name)
        
        return sorted(decorators)
    
    def search(self, query: str) -> List[str]:
        """Search decorators by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for name, metadata in self._decorators.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue
            
            # Search in description
            if query_lower in metadata.description.lower():
                results.append(name)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                results.append(name)
                continue
        
        return sorted(results)
    
    def unregister(self, name: str) -> bool:
        """Unregister a decorator."""
        try:
            if name in self._decorators:
                metadata = self._decorators[name]
                
                # Remove from categories
                if name in self._categories[metadata.category]:
                    self._categories[metadata.category].remove(name)
                
                # Remove aliases
                aliases_to_remove = [alias for alias, actual_name in self._aliases.items() if actual_name == name]
                for alias in aliases_to_remove:
                    del self._aliases[alias]
                
                # Remove decorator
                del self._decorators[name]
                
                self.logger.info(f"Decorator {name} unregistered successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unregister decorator {name}: {e}")
            return False
    
    def update_metadata(
        self, 
        name: str, 
        **updates
    ) -> bool:
        """Update metadata for a decorator."""
        try:
            if name not in self._decorators:
                return False
            
            metadata = self._decorators[name]
            
            # Update allowed fields
            allowed_fields = [
                'description', 'tags', 'deprecated', 'category', 
                'migration_target', 'dependencies', 'performance_impact', 'security_level'
            ]
            
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(metadata, field, value)
                    
                    # Update category mapping if category changed
                    if field == 'category':
                        # Remove from old category
                        if name in self._categories[metadata.category]:
                            self._categories[metadata.category].remove(name)
                        # Add to new category
                        self._categories[value].append(name)
            
            self.logger.info(f"Metadata updated for decorator {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata for decorator {name}: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all decorators."""
        stats = {
            "total_decorators": len(self._decorators),
            "total_registrations": self._total_registrations,
            "total_usage": self._total_usage,
            "categories": {cat.value: len(names) for cat, names in self._categories.items()},
            "deprecated_count": len([d for d in self._decorators.values() if d.deprecated]),
            "most_used": [],
            "recently_used": [],
            "by_category": {}
        }
        
        # Most used decorators
        sorted_by_usage = sorted(
            self._decorators.values(), 
            key=lambda x: x.usage_count, 
            reverse=True
        )
        stats["most_used"] = [
            {"name": d.name, "usage_count": d.usage_count} 
            for d in sorted_by_usage[:10]
        ]
        
        # Recently used decorators
        recently_used = [d for d in self._decorators.values() if d.last_used]
        sorted_by_recent = sorted(
            recently_used, 
            key=lambda x: x.last_used, 
            reverse=True
        )
        stats["recently_used"] = [
            {"name": d.name, "last_used": d.last_used.isoformat()} 
            for d in sorted_by_recent[:10]
        ]
        
        # Statistics by category
        for category, names in self._categories.items():
            category_decorators = [self._decorators[name] for name in names if name in self._decorators]
            stats["by_category"][category.value] = {
                "count": len(category_decorators),
                "total_usage": sum(d.usage_count for d in category_decorators),
                "deprecated_count": len([d for d in category_decorators if d.deprecated])
            }
        
        return stats
    
    def export_registry(self, file_path: str, format: str = "json") -> bool:
        """Export the registry to a file."""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_decorators": len(self._decorators),
                "decorators": {}
            }
            
            for name, metadata in self._decorators.items():
                export_data["decorators"][name] = metadata.to_dict()
            
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Registry exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, file_path: str, format: str = "json", overwrite: bool = False) -> bool:
        """Import decorators from a file."""
        try:
            if format.lower() == "json":
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            imported_count = 0
            for name, decorator_data in import_data.get("decorators", {}).items():
                # Skip if decorator already exists and overwrite is False
                if name in self._decorators and not overwrite:
                    continue
                
                # Note: We can't import the actual decorator function, only metadata
                # The decorator would need to be re-registered
                self.logger.warning(f"Skipping decorator {name} - decorator functions cannot be imported from files")
                imported_count += 1
            
            self.logger.info(f"Registry import completed. {imported_count} decorators processed.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import registry: {e}")
            return False
    
    def cleanup(self, max_age_days: int = 30) -> int:
        """Clean up old unused decorators."""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            decorators_to_remove = []
            for name, metadata in self._decorators.items():
                if (metadata.deprecated and 
                    metadata.last_used and 
                    metadata.last_used < cutoff_date and
                    metadata.usage_count == 0):
                    decorators_to_remove.append(name)
            
            for name in decorators_to_remove:
                if self.unregister(name):
                    cleaned_count += 1
            
            self._last_cleanup = datetime.now()
            self.logger.info(f"Cleanup completed. {cleaned_count} decorators removed.")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_migration_paths(self) -> Dict[str, str]:
        """Get migration paths for deprecated decorators."""
        migration_paths = {}
        for name, metadata in self._decorators.items():
            if metadata.deprecated and metadata.migration_target:
                migration_paths[name] = metadata.migration_target
        return migration_paths
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate decorator dependencies."""
        missing_deps = {}
        for name, metadata in self._decorators.items():
            missing = []
            for dep in metadata.dependencies:
                if dep not in self._decorators:
                    missing.append(dep)
            if missing:
                missing_deps[name] = missing
        return missing_deps
    
    def __len__(self) -> int:
        """Return the number of registered decorators."""
        return len(self._decorators)
    
    def __contains__(self, name: str) -> bool:
        """Check if a decorator is registered."""
        return name in self._decorators or name in self._aliases
    
    def __iter__(self):
        """Iterate over decorator names."""
        return iter(self._decorators.keys())

# Global registry instance
decorator_registry = DecoratorRegistry()

# Import timedelta for cleanup method
from datetime import timedelta