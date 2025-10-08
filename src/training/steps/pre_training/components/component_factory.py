"""Component factory and registry utilities for pre-training pipeline components."""

from __future__ import annotations

import difflib
import importlib
import os
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Type

try:  # Python 3.10+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for older Python versions
    import importlib_metadata  # type: ignore

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import validate_finite
from src.utils.serialization_utils import JSONSerializer
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    safe_matrix_multiply,
    optimize_dataframe
)
from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging

from .base_component import BasePreTrainingComponent, ComponentConfig

logger = system_logger.getChild('PreTrainingComponentFactory')
factory_event_logger = PreTrainingEventLogger(configure_pre_training_logging())


def _get_component_typo_suggestions(component_name: str, available_components: list[str]) -> str:
    """Generate typo suggestions for unknown component names."""
    if not available_components:
        return ""

    # Get close matches using difflib
    close_matches = difflib.get_close_matches(
        component_name,
        available_components,
        n=3,
        cutoff=0.6  # Only suggest if similarity is at least 60%
    )

    if close_matches:
        suggestion_text = "Did you mean one of these?"
        for match in close_matches:
            suggestion_text += f"\n  - '{match}'"
        return suggestion_text

    return ""


def _factory_context(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        'step': 'component_factory',
        'run_id': None,
        'symbol': None,
        'timeframe': None,
        'rows_in': None,
        'rows_out': None,
    }
    if extra:
        payload.update(extra)
    return payload


def _log_info(message: str, **context: Any) -> None:
    payload = _factory_context(context)
    logger.info(message)
    factory_event_logger.info(message, context=payload)


def _log_warning(message: str, **context: Any) -> None:
    payload = _factory_context(context)
    logger.warning(message)
    factory_event_logger.warning(message, context=payload)


def _log_error(message: str, **context: Any) -> None:
    payload = _factory_context(context)
    logger.error(message)
    factory_event_logger.error(message, context=payload)


def _log_debug(message: str, **context: Any) -> None:
    payload = _factory_context(context)
    logger.debug(message)
    factory_event_logger.info(message, context=payload)


@dataclass
class ComponentRegistration:
    """Container describing a registered pre-training component."""

    name: str
    component_class: Optional[Type[BasePreTrainingComponent]]
    available: bool = True
    error: Optional[str] = None
    extras: Optional[str] = None
    source: Optional[str] = None

    def instantiate(self, config: Optional[ComponentConfig]) -> BasePreTrainingComponent:
        """Instantiate the registered component."""

        if not self.available or self.component_class is None:
            error_message = self.error or "Component is not available."
            raise ValueError(
                f"Component {self.name} is not available. {error_message}"
            )
        return self.component_class(config)


class ComponentRegistry:
    """Runtime registry of pre-training pipeline components."""

    def __init__(self) -> None:
        self._components: Dict[str, ComponentRegistration] = {}
        self._loaded_entry_point_groups: Set[str] = set()

    def register(
        self,
        name: str,
        component_class: Optional[Type[BasePreTrainingComponent]],
        *,
        available: bool = True,
        error: Optional[str] = None,
        extras: Optional[str] = None,
        source: Optional[str] = None,
        override: bool = False,
    ) -> ComponentRegistration:
        """Register a component in the registry."""

        # Check if component is already registered
        if name in self._components and not override:
            existing = self._components[name]
            # Allow re-registration if it's the same component class (handles circular imports)
            # Check both identity and qualified name to handle circular imports
            if (existing.component_class is component_class or 
                (existing.component_class is not None and component_class is not None and
                 f"{existing.component_class.__module__}.{existing.component_class.__name__}" ==
                 f"{component_class.__module__}.{component_class.__name__}")):
                # Already registered with same class, just return existing registration
                return existing
            # Different component class trying to use same name - log warning but allow it
            # (some components may be registered multiple times during development)
            import logging
            logging.getLogger(__name__).warning(
                f"Component '{name}' is already registered. "
                f"Existing: {existing.component_class}, New: {component_class}. "
                f"Returning existing registration."
            )
            return existing

        if available and component_class is not None and not issubclass(
            component_class, BasePreTrainingComponent
        ):
            raise ValueError(
                "Component class must inherit from BasePreTrainingComponent"
            )

        registration = ComponentRegistration(
            name=name,
            component_class=component_class,
            available=available and component_class is not None,
            error=error,
            extras=extras,
            source=source or (component_class.__module__ if component_class else None),
        )
        self._components[name] = registration
        status = "available" if registration.available else "unavailable"

        # Log registration with additional utility context
        _log_debug(
            f"🧾 [PRE_TRAINING_FACTORY] Registered component '{name}' ({status})",
            component=name,
            available=registration.available,
            source=registration.source,
        )

        # Validate component has required utility methods if available
        if registration.available and registration.component_class:
            try:
                tprint(f"🔍 Validating utility methods for component '{name}'")

                # Check if component has utility methods (non-blocking validation)
                required_methods = ['safe_dataframe_operation', 'validate_finite_values', 'get_memory_pressure']
                missing_methods = []
                for method in required_methods:
                    if not hasattr(registration.component_class, method):
                        missing_methods.append(method)

                if missing_methods:
                    tprint_warning(f"⚠️ Component '{name}' missing utility methods: {missing_methods}")
                    _log_warning(
                        f"⚠️ [PRE_TRAINING_FACTORY] Component '{name}' missing utility methods: {missing_methods}",
                        component=name,
                        missing_methods=missing_methods,
                    )
                else:
                    tprint_success(f"✅ Component '{name}' has all required utility methods")
                    _log_debug(
                        f"✅ [PRE_TRAINING_FACTORY] Component '{name}' has all required utility methods",
                        component=name,
                    )

                # Check for matrix operation methods
                matrix_methods = ['safe_matrix_multiply', 'optimize_dataframe_for_matrix_ops']
                missing_matrix_methods = []
                for method in matrix_methods:
                    if not hasattr(registration.component_class, method):
                        missing_matrix_methods.append(method)

                if missing_matrix_methods:
                    tprint_warning(f"⚠️ Component '{name}' missing matrix operation methods: {missing_matrix_methods}")
                    _log_warning(
                        f"⚠️ [PRE_TRAINING_FACTORY] Component '{name}' missing matrix operation methods: {missing_matrix_methods}",
                        component=name,
                        missing_matrix_methods=missing_matrix_methods,
                    )
                else:
                    tprint_success(f"✅ Component '{name}' has all required matrix operation methods")

            except Exception as e:
                tprint_error(f"❌ Could not validate methods for component '{name}': {e}")
                _log_warning(
                    f"⚠️ [PRE_TRAINING_FACTORY] Could not validate utility methods for '{name}': {e}",
                    component=name,
                    error=str(e),
                )

        return registration

    def mark_unavailable(
        self,
        name: str,
        *,
        error: Optional[str] = None,
        extras: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ComponentRegistration:
        """Register an unavailable component placeholder."""

        return self.register(
            name,
            component_class=None,
            available=False,
            error=error,
            extras=extras,
            source=source,
            override=True,
        )

    def get(self, name: str) -> Optional[ComponentRegistration]:
        """Retrieve a component registration by name."""

        return self._components.get(name)

    def unregister(self, name: str) -> None:
        """Remove a registration (used for testing)."""

        self._components.pop(name, None)

    def available_components(self) -> Dict[str, ComponentRegistration]:
        """Return a mapping of available components."""

        return {
            name: registration
            for name, registration in self._components.items()
            if registration.available
        }

    def all_components(self) -> Dict[str, ComponentRegistration]:
        """Return all registered components including unavailable ones."""

        return dict(self._components)

    def load_entry_points(self, group: str) -> None:
        """Load additional component registrations from entry points."""

        if group in self._loaded_entry_point_groups:
            return

        self._loaded_entry_point_groups.add(group)

        try:
            entry_points = importlib_metadata.entry_points()
        except Exception as exc:  # pragma: no cover - very defensive
            _log_warning(
                f"⚠️ [PRE_TRAINING_FACTORY] Could not load entry points: {exc}",
                group=group,
                error=str(exc),
            )
            return

        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=group)  # type: ignore[attr-defined]
        else:  # pragma: no cover - Python <3.10
            selected = [
                ep for ep in entry_points  # type: ignore[assignment]
                if getattr(ep, "group", None) == group
            ]

        for entry_point in selected:
            try:
                loaded = entry_point.load()
            except Exception as exc:
                self.mark_unavailable(
                    entry_point.name,
                    error=str(exc),
                    extras=getattr(entry_point, "value", None),
                    source=getattr(entry_point, "module", None),
                )
                continue

            if (
                isinstance(loaded, type)
                and issubclass(loaded, BasePreTrainingComponent)
                and entry_point.name not in self._components
            ):
                self.register(
                    entry_point.name,
                    loaded,
                    extras=getattr(entry_point, "value", None),
                    source=loaded.__module__,
                )


_registry = ComponentRegistry()


def register_component(
    name: str,
    *,
    extras: Optional[str] = None,
    available: bool = True,
    error: Optional[str] = None,
    override: bool = False,
):
    """Decorator used by components to register themselves with the factory."""

    def decorator(component_class: Type[BasePreTrainingComponent]):
        if not available:
            _registry.mark_unavailable(
                name,
                error=error,
                extras=extras,
                source=component_class.__module__,
            )
            return component_class

        _registry.register(
            name,
            component_class,
            extras=extras,
            override=override,
        )
        return component_class

    return decorator


def register_unavailable_component(
    name: str,
    *,
    error: Optional[str] = None,
    extras: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Register a component placeholder when optional dependencies are missing."""

    _registry.mark_unavailable(name, error=error, extras=extras, source=source)


class ComponentFactory:
    """Factory for creating pre-training pipeline components."""

    registry: ComponentRegistry = _registry

    _initialized: bool = False
    _aliases_registered: bool = False

    # Initialize utility managers for the factory
    _common_utils = CommonUtilities()
    _json_serializer = JSONSerializer()
    ENTRY_POINT_GROUP = "ares.pre_training.components"
    EXTRA_MODULES_ENV = "ARES_PRETRAIN_COMPONENT_MODULES"
    EXTRA_MODULES_FILE = (
        Path(__file__).resolve().parents[4]
        / "config"
        / "pre_training_component_modules.txt"
    )
    BUILTIN_MODULES = (
        "src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization",
        "src.training.steps.pre_training.components.final_feature_selection",
        "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.optimized_lookback_component",
        "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component",
        # Removed multi_horizon_component - replaced with analyst_profit_labeler and tactician_entry_labeler
        "src.training.steps.pre_training.analyst_profit_labeler",
        "src.training.steps.pre_training.tactician_entry_labeler",
    )
    DEFAULT_ALIASES = {
        # Tactician orchestrator specific aliases to provide clearer diagnostics
        "tactician_feature_optimization": "feature_lookback_optimization",
        "tactician_feature_generation": "interactive_feature_generation",
        "tactician_horizon_labeling": "tactician_entry_labeler",  # Changed from multi_horizon_profit_labeler
        "tactician_feature_selection": "final_feature_selection",
        # Role-specific aliases for feature engineering
        "analyst_feature_lookback_optimization": "feature_lookback_optimization",
        "analyst_interactive_feature_generation": "interactive_feature_generation",
        "analyst_final_feature_selection": "final_feature_selection",
        "tactician_feature_lookback_optimization": "feature_lookback_optimization",
        "tactician_interactive_feature_generation": "interactive_feature_generation",
        "tactician_final_feature_selection": "final_feature_selection",
    }

    @classmethod
    def create_component(
        cls,
        component_name: str,
        config: Optional[ComponentConfig] = None,
    ) -> BasePreTrainingComponent:
        """Create and return a registered component instance."""

        _log_info(
            f"🏭 [PRE_TRAINING_FACTORY] Creating component: {component_name}",
            event='component_factory_create',
            component=component_name,
        )

        cls._ensure_initialized()

        registration = cls.registry.get(component_name)
        if registration is None:
            available_components = list(cls.registry.available_components().keys())
            typo_suggestions = _get_component_typo_suggestions(component_name, available_components)

            _log_error(
                f"❌ [PRE_TRAINING_FACTORY] Unknown component: {component_name}",
                event='component_factory_unknown',
                component=component_name,
            )
            _log_info(
                f"📊 [PRE_TRAINING_FACTORY] Available components: {available_components}",
                event='component_factory_available_list',
                available_components=available_components,
            )

            error_message = f"Unknown component: '{component_name}'. "
            error_message += f"This component is not registered in the pre-training factory.\n\n"
            error_message += f"Available components ({len(available_components)} total):\n"
            error_message += f"  {', '.join(sorted(available_components))}\n\n"

            if typo_suggestions:
                error_message += f"{typo_suggestions}\n\n"

            error_message += "Component registration troubleshooting:\n"
            error_message += "- Check if the component name is spelled correctly\n"
            error_message += "- Verify the component is imported in the __init__.py file\n"
            error_message += "- Ensure the component inherits from BasePreTrainingComponent\n"
            error_message += "- Check if the component has a proper @component_config decorator\n"

            # Try to serialize error details for debugging
            error_details = {
                'component_name': component_name,
                'available_components': available_components,
                'typo_suggestions': typo_suggestions,
                'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None,
            }
            try:
                cls._json_serializer.save(error_details, f"/tmp/factory_error_{component_name}.json")
            except Exception as e:
                _log_warning(f"⚠️ Could not save error details: {e}")

            raise ValueError(error_message)

        _log_info(
            f"🔧 [PRE_TRAINING_FACTORY] Creating {component_name} from registered components",
            event='component_factory_instantiate',
            component=component_name,
        )

        try:
            component = registration.instantiate(config)
            # Validate component has required utility methods
            if hasattr(component, 'safe_dataframe_operation'):
                _log_debug(
                    f"✅ [PRE_TRAINING_FACTORY] Component {component_name} has utility methods",
                    component=component_name,
                )
        except ValueError as exc:
            _log_error(
                f"❌ [PRE_TRAINING_FACTORY] Component {component_name} is not available: {exc}",
                event='component_factory_unavailable',
                component=component_name,
                error=str(exc),
            )
            raise

        _log_info(
            f"✅ [PRE_TRAINING_FACTORY] Successfully created {component_name}",
            event='component_factory_created',
            component=component_name,
        )
        return component

    @classmethod
    def register_component(
        cls,
        name: str,
        component_class: Type[BasePreTrainingComponent],
    ) -> None:
        """Register a component class with the factory."""

        cls.registry.register(name, component_class)
        _log_info(
            f"🧾 [PRE_TRAINING_FACTORY] Registered component: {name}",
            event='component_factory_register',
            component=name,
        )

    @classmethod
    def get_available_components(cls) -> list[str]:
        """Return a list of available component names."""

        cls._ensure_initialized()
        available = list(cls.registry.available_components().keys())
        _log_info(
            f"📋 [PRE_TRAINING_FACTORY] Available components: {available}",
            event='component_factory_available_components',
            available_components=available,
        )
        return available

    @classmethod
    def is_component_available(cls, component_name: str) -> bool:
        """Return True when the requested component can be instantiated."""

        cls._ensure_initialized()
        registration = cls.registry.get(component_name)
        available = bool(registration and registration.available)
        _log_info(
            f"🔍 [PRE_TRAINING_FACTORY] Component '{component_name}' available: {available}",
            event='component_factory_availability',
            component=component_name,
            available=available,
        )
        return available

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure that built-in registrations and extras have been loaded."""

        if cls._initialized:
            return

        for module_path in cls.BUILTIN_MODULES:
            cls._import_module(module_path)

        for module_path in cls._load_extra_modules_from_env():
            cls._import_module(module_path)

        for module_path in cls._load_extra_modules_from_file():
            cls._import_module(module_path)

        cls.registry.load_entry_points(cls.ENTRY_POINT_GROUP)

        # Manually register components that don't auto-register due to import issues
        cls._register_missing_components()

        cls._register_default_aliases()

        cls._initialized = True
        cls._aliases_registered = True
    
    @classmethod
    def _register_missing_components(cls) -> None:
        """Register components that exist but aren't auto-registered."""
        try:
            # Try to register final_feature_selection
            try:
                from .final_feature_selection import FinalFeatureSelectionComponent
                if 'final_feature_selection' not in cls.registry.available_components():
                    cls.registry.register('final_feature_selection', FinalFeatureSelectionComponent)
                    _log_info("✅ Manually registered final_feature_selection")
            except Exception as e:
                _log_warning(f"⚠️ Could not register final_feature_selection: {e}")
            
            # Try to register feature_lookback_optimization
            try:
                from ..feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimizationComponent
                if 'feature_lookback_optimization' not in cls.registry.available_components():
                    cls.registry.register('feature_lookback_optimization', FeatureLookbackOptimizationComponent)
                    _log_info("✅ Manually registered feature_lookback_optimization")
            except Exception as e:
                _log_warning(f"⚠️ Could not register feature_lookback_optimization: {e}")
            
            # Try to register interactive_feature_generation
            try:
                from ..interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import InteractiveFeatureGenerationComponent
                if 'interactive_feature_generation' not in cls.registry.available_components():
                    cls.registry.register('interactive_feature_generation', InteractiveFeatureGenerationComponent)
                    _log_info("✅ Manually registered interactive_feature_generation")
            except Exception as e:
                _log_warning(f"⚠️ Could not register interactive_feature_generation: {e}")
        except Exception as e:
            _log_warning(f"⚠️ Error during manual component registration: {e}")

    @classmethod
    def register_alias(cls, alias: str, target: str) -> None:
        """Register an alias that points at an existing component registration."""

        cls._ensure_initialized()
        cls._register_alias(alias, target)

    @classmethod
    def _register_default_aliases(cls) -> None:
        """Register built-in aliases used by higher-level orchestrators."""

        if cls._aliases_registered:
            return

        for alias, target in cls.DEFAULT_ALIASES.items():
            cls._register_alias(alias, target)

    @classmethod
    def _register_alias(cls, alias: str, target: str) -> None:
        """Internal helper to register or update a component alias."""

        if not target:
            _log_warning(
                f"⚠️ [PRE_TRAINING_FACTORY] Cannot register alias '{alias}' without a target",
                event="component_factory_alias_missing_target",
                alias=alias,
            )
            return

        registration = cls.registry.get(target)
        if registration is None:
            _log_warning(
                f"⚠️ [PRE_TRAINING_FACTORY] Alias target '{target}' not registered; '{alias}' marked unavailable",
                event="component_factory_alias_missing_target",
                alias=alias,
                target=target,
            )
            cls.registry.mark_unavailable(
                alias,
                error=f"Alias target '{target}' not registered",
                source="component_factory.alias",
            )
            return

        cls.registry.register(
            alias,
            registration.component_class,
            available=registration.available,
            error=registration.error,
            extras=registration.extras,
            source=registration.source,
            override=True,
        )

    @classmethod
    def _import_module(cls, module_path: str) -> None:
        """Import a module and log failures as warnings."""

        if not module_path:
            return

        try:
            importlib.import_module(module_path)
            _log_debug(
                f"📦 [PRE_TRAINING_FACTORY] Imported component module '{module_path}'",
                module=module_path,
                event='component_factory_module_imported',
            )
        except Exception as exc:
            _log_warning(
                f"⚠️ [PRE_TRAINING_FACTORY] Could not import '{module_path}': {exc}",
                module=module_path,
                error=str(exc),
                event='component_factory_module_import_failed',
            )

    @classmethod
    def _load_extra_modules_from_env(cls) -> Iterable[str]:
        """Load additional component modules defined in environment variables."""

        modules = os.environ.get(cls.EXTRA_MODULES_ENV)
        if not modules:
            return []

        return [module.strip() for module in modules.split(",") if module.strip()]

    @classmethod
    def _load_extra_modules_from_file(cls) -> Iterable[str]:
        """Load extra modules from an optional configuration file."""

        if not cls.EXTRA_MODULES_FILE.exists():
            return []

        try:
            with open(cls.EXTRA_MODULES_FILE, "r", encoding="utf-8") as handle:
                lines = [
                    line.strip()
                    for line in handle.readlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            _log_warning(
                f"⚠️ [PRE_TRAINING_FACTORY] Failed to read extra modules file: {exc}",
                file=str(cls.EXTRA_MODULES_FILE),
                error=str(exc),
                event='component_factory_module_file_failed',
            )
            return []

        return lines

    @classmethod
    def _unregister_for_testing(cls, name: str) -> None:
        """Remove a component registration (testing helper)."""

        cls.registry.unregister(name)
