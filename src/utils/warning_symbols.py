"""Warning symbols and constants.

Provide both string symbols and callable helpers expected by logger and other modules.
"""

# Symbol constants
warning = "⚠️"
invalid = "❌"
missing = "❓"
error = "⛔"
failed = "💥"
critical = "🔥"

# Callable helpers used by logger fallback pattern
def _print_with(label: str, msg: object) -> None:
    try:
        print(f"{label}: {msg}")
    except Exception:
        print(f"{label}")

def warning(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("WARNING", msg)

def error(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("ERROR", msg)

def failed(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("FAILED", msg)

def critical(msg: object) -> None:  # type: ignore[func-assign]
    _print_with("CRITICAL", msg)