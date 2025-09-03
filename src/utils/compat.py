"""Compatibility shims to forward legacy decorators to core equivalents.

This module allows gradual migration by re-exporting adapter functions
that map old decorator signatures to the new core decorators.
"""

from collections.abc import Callable
from typing import Any

from src.core.decorators import handles_errors as _handles_errors


def handle_errors(
	*,
	exceptions: tuple[type[Exception], ...] = (Exception,),
	default_return: Any = None,
	**kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
	"""Adapter for legacy handle_errors to core handles_errors.

	Maps exceptions to varargs and default_return to fallback.
	Ignores unsupported legacy kwargs (e.g., context).
	"""
	return _handles_errors(*exceptions, fallback=default_return)


def handle_specific_errors(
	*,
	error_handlers: dict[type[Exception], tuple[Any, str]] | None = None,
	default_return: Any = None,
	**kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
	"""Adapter for legacy handle_specific_errors with per-exception return values.

	If an exception matches a key in error_handlers, returns the mapped
	return value. Otherwise returns default_return (or re-raises if
	kwargs contains reraise=True).
	"""

	def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
		import asyncio
		from functools import wraps

		if asyncio.iscoroutinefunction(func):
			@wraps(func)
			async def async_wrapper(*args: Any, **kw: Any) -> Any:
				try:
					return await func(*args, **kw)
				except Exception as exc:  # noqa: BLE001
					if error_handlers:
						for exc_type, (return_value, _msg) in error_handlers.items():
							if isinstance(exc, exc_type):
								return return_value
					if kwargs.get("reraise"):
						raise
					return default_return

			return async_wrapper
		@wraps(func)
		def sync_wrapper(*args: Any, **kw: Any) -> Any:
			try:
				return func(*args, **kw)
			except Exception as exc:  # noqa: BLE001
				if error_handlers:
					for exc_type, (return_value, _msg) in error_handlers.items():
						if isinstance(exc, exc_type):
							return return_value
				if kwargs.get("reraise"):
					raise
				return default_return

		return sync_wrapper

	return decorator

