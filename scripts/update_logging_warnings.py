#!/usr/bin/env python3
"""
Script to update logging messages in training step files with warning symbols.

This script automatically adds warning symbols to error and warning messages
throughout the training step files to make issues more visible.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

from src.utils.warning_symbols import missing, warning

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_warning_symbol_function(message: str) -> str:
    """
    Determine the appropriate warning symbol function based on the message content.

    Args:
        message: The error/warning message

    Returns:
        The appropriate warning symbol function name
    """
    message_lower = message.lower()

    # Error patterns
    if any(word in message_lower for word in ["failed", "failure", "fail"]):
        return "failed"
    if any(word in message_lower for word in ["invalid", "invalid configuration"]):
        return "invalid"
    if any(
        word in message_lower for word in ["missing", "not found", "file not found"]
    ):
        return "missing"
    if any(word in message_lower for word in ["timeout", "timed out"]):
        return "timeout"
    if any(word in message_lower for word in ["connection", "network"]):
        return "connection_error"
    if any(word in message_lower for word in ["validation", "validate"]):
        return "validation_error"
    if any(word in message_lower for word in ["initialization", "init", "initialize"]):
        return "initialization_error"
    if any(word in message_lower for word in ["execution", "execute", "runtime"]):
        return "execution_error"
    if any(word in message_lower for word in ["critical", "fatal"]):
        return "critical"
    if any(word in message_lower for word in ["problem", "issue"]):
        return "problem"
    # Default to error for error messages, warning for warning messages
    return "error"


def update_file_logging_messages(file_path: str) -> tuple[int, int]:
    """
    Update logging messages in a file with warning symbols.

    Args:
        file_path: Path to the file to update

    Returns:
        Tuple of (number of changes made, number of lines processed)
    """
    changes_made = 0

    try:
        path_obj = Path(file_path)
        with path_obj.open(encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Replace logger.* calls
        logger_pattern = re.compile(
            r"logger\.(error|warning|exception|critical)\((?:f)?([\"\'])(.*?)(?:\2)\)",
            re.DOTALL,
        )

        def replace_logger(match: re.Match[str]) -> str:
            nonlocal changes_made
            method = match.group(1)
            message = match.group(3)
            warning_func = get_warning_symbol_function(message)
            changes_made += 1
            return f'logger.{method}({warning_func}("{message}"))'

        content = logger_pattern.sub(replace_logger, content)

        # Replace print statements starting with emojis
        print_pattern = re.compile(
            r"print\((?:f)?([\"\'])((?:❌|⚠️|🚨) )?(.*?)(?:\1)\)",
            re.DOTALL,
        )

        def replace_print(match: re.Match[str]) -> str:
            nonlocal changes_made
            emoji = match.group(2) or ""
            message = match.group(3)
            if "❌" in emoji or "🚨" in emoji:
                warning_func = "error"
            elif "⚠️" in emoji:
                warning_func = "warning"
            else:
                warning_func = get_warning_symbol_function(message)
            changes_made += 1
            return f'print({warning_func}("{message}"))'

        content = print_pattern.sub(replace_print, content)

        # Only write if changes were made
        if content != original_content:
            with path_obj.open("w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Updated {file_path} with {changes_made} changes")
        else:
            print(f"ℹ️  No changes needed for {file_path}")

        return changes_made, len(content.split("\n"))

    except Exception as e:  # noqa: BLE001
        print(warning(f"Error processing {file_path}: {e}"))
        return 0, 0


def add_warning_symbols_import(file_path: str) -> bool:
    """
    Add warning symbols import to a file if it doesn't already have it.

    Args:
        file_path: Path to the file to update

    Returns:
        True if import was added, False otherwise
    """
    try:
        path_obj = Path(file_path)
        with path_obj.open(encoding="utf-8") as f:
            content = f.read()

        # Check if warning symbols are already imported
        if "from src.utils.warning_symbols import" in content:
            return False

        # Build import block
        warning_import = (
            "from src.utils.warning_symbols import ("
            "error, warning, critical, problem, failed, invalid, missing, timeout, "
            "connection_error, validation_error, initialization_error, execution_error)"
        )

        # Find the logger import line
        logger_import_pattern = r"from src\.utils\.logger import.*"
        match = re.search(logger_import_pattern, content)

        if match:
            # Add warning symbols import after logger import
            new_content = content.replace(
                match.group(0), match.group(0) + "\n" + warning_import,
            )
        else:
            # Prepend import at the top if logger import not found
            new_content = warning_import + "\n" + content

        with path_obj.open("w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"✅ Added warning symbols import to {file_path}")
        return True
    except Exception as e:  # noqa: BLE001
        print(warning(f"Error adding import to {file_path}: {e}"))
        return False


def main() -> None:
    """Main function to update all training step files."""
    training_steps_dir = project_root / "src" / "training" / "steps"

    if not training_steps_dir.exists():
        print(missing(f"Training steps directory not found: {training_steps_dir}"))
        return

    # Get all Python files in the training steps directory
    python_files = list(training_steps_dir.glob("*.py"))

    print(f"🔍 Found {len(python_files)} Python files in training steps directory")

    total_changes = 0
    total_files_processed = 0

    for file_path in python_files:
        print(f"\n📁 Processing {file_path.name}...")

        # Add warning symbols import if needed
        import_added = add_warning_symbols_import(str(file_path))

        # Update logging messages
        changes, _lines = update_file_logging_messages(str(file_path))

        total_changes += changes
        if import_added:
            total_changes += 1
        total_files_processed += 1

    print("\n✅ Summary:")
    print(f"   Files processed: {total_files_processed}")
    print(f"   Total changes made: {total_changes}")
    avg = total_changes / total_files_processed if total_files_processed else 0
    print(f"   Average changes per file: {avg:.1f}")


if __name__ == "__main__":
    main()
