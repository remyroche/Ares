"""Component factory and registry utilities for pre-training pipeline components."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Type

try:  # Python 3.10+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for older Python versions
    import importlib_metadata  # type: ignore

from src.utils.logger import system_logger
from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging

from .base_component import BasePreTrainingComponent, ComponentConfig

logger = system_logger.getChild('PreTrainingComponentFactory')
factory_event_logger = PreTrainingEventLogger(configure_pre_training_logging())


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

        if name in self._components and not override:
            raise ValueError(f"Component '{name}' is already registered")

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
        _log_debug(
            f"🧾 [PRE_TRAINING_FACTORY] Registered component '{name}' ({status})",
            component=name,
            available=registration.available,
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
        "src.training.steps.pre_training.components.multi_horizon_component",
        "src.training.steps.pre_training.components.pid_based_feature_generation_registration",
        "src.training.steps.pre_training.analyst_profit_labeler",
        "src.training.steps.pre_training.tactician_entry_labeler",
    )
    DEFAULT_ALIASES = {
        # Tactician orchestrator specific aliases to provide clearer diagnostics
        "tactician_feature_optimization": "feature_lookback_optimization",
        "tactician_pid_generation": "pid_based_feature_generation",
        "tactician_horizon_labeling": "multi_horizon_profit_labeler",
        "tactician_feature_selection": "final_feature_selection",
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
            raise ValueError(
                f"Unknown component: {component_name}. Available components: {available_components}"
            )

        _log_info(
            f"🔧 [PRE_TRAINING_FACTORY] Creating {component_name} from registered components",
            event='component_factory_instantiate',
            component=component_name,
        )

        try:
            component = registration.instantiate(config)
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

        cls._register_default_aliases()

        cls._initialized = True
        cls._aliases_registered = True

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
