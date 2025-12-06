from typing import Any, Callable, Awaitable

# Minimal typed interfaces and helpers for non-trading contexts

def tprint(*args, **kwargs):
    return None

def handle_errors(fn: Callable = None, **kwargs):
    # Support decorator usage with optional arguments (e.g., default_return=...).
    if fn is None:
        def wrapper(f: Callable):
            return f
        return wrapper
    return fn

def handle_async_errors(fn: Callable[..., Awaitable[Any]] = None, **kwargs):
    # Support decorator usage with optional arguments (e.g., default_return=...).
    if fn is None:
        def wrapper(f: Callable[..., Awaitable[Any]]):
            return f
        return wrapper
    return fn

class DataSource:
    def __init__(self, *args, **kwargs):
        pass

class ValidationResult:
    def __init__(self, *args, **kwargs):
        self.valid = True
        self.errors = []

class IHighLevelAuthManager: ...
class IHighLevelMarketManager: ...
class IHighLevelOrderManager: ...
class IHighLevelRiskManager: ...
class IHighLevelBalanceManager: ...
class IHighLevelRateLimitManager: ...
