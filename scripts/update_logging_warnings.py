#!/usr/bin/env python3
"""
Script to update logging messages in training step files with warning symbols.

This script automatically adds warning symbols to error and warning messages
throughout the training step files to make issues more visible.
"""

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
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Pattern to match logger.error, logger.warning, logger.exception, logger.critical calls
        # Also match print statements with error/warning indicators
        patterns = [
            # logger.*("message") -> wrap with symbol function
            (r'logger\.(error|warning|exception|critical)\(f?"([^"]*)"', "logger"),
            # print emoji patterns -> replace with print(symbol("message"))
            (r'print\(f?"❌ ([^"]*)"', "failed"),
            (r'print\(f?"⚠️ ([^"]*)"', "warning"),
            (r'print\(f?"🚨 ([^"]*)"', "error"),
        ]

        for pattern, kind in patterns:
            matches = list(re.finditer(pattern, content))
            for match in reversed(matches):
                if kind == "logger":
                    method = match.group(1)
                    message = match.group(2)
                    sym = get_warning_symbol_function(message)
                    replacement = f'logger.{method}({sym}("{message}"))'
                else:
                    message = match.group(1)
                    sym = kind
                    replacement = f'print({sym}("{message}"))'

                start, end = match.span()
                content = content[:start] + replacement + content[end:]
                changes_made += 1

        # Only write if changes were made
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
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
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Check if warning symbols are already imported
        if "from src.utils.warning_symbols import" in content:
            return False

        # Find the logger import line
        logger_import_pattern = r"from src\.utils\.logger import.*"
        match = re.search(logger_import_pattern, content)

        if match:
            # Add warning symbols import after logger import
            warning_import = (
                "from src.utils.warning_symbols import (\n"
                "    error,\n"
                "    warning,\n"
                "    problem,\n"
                "    failed,\n"
                "    missing,\n"
                "    timeout,\n"
                "    connection_error,\n"
                "    validation_error,\n"
                "    initialization_error,\n"
                "    execution_error\n"
                ")"
            )

            # Insert after the logger import
            new_content = content.replace(
                match.group(0),
                match.group(0) + "\n" + warning_import,
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"✅ Added warning symbols import to {file_path}")
            return True

        # Try to find any import line to add after
        import_pattern = r"^import .*$|^from .* import .*$"
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if re.match(import_pattern, line.strip()):
                # Add warning symbols import after this import
                warning_import = (
                    "from src.utils.warning_symbols import (\n"
                    "    error,\n"
                    "    warning,\n"
                    "    problem,\n"
                    "    failed,\n"
                    "    missing,\n"
                    "    timeout,\n"
                    "    connection_error,\n"
                    "    validation_error,\n"
                    "    initialization_error,\n"
                    "    execution_error\n"
                    ")"
                )

                lines.insert(i + 1, warning_import)
                new_content = "\n".join(lines)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                print(f"✅ Added warning symbols import to {file_path}")
                return True

        print(f" Could not find suitable import location in {file_path}")
        return False

    except Exception as e:  # noqa: BLE001
        print(f"Error adding import to {file_path}: {e}")
        return False

if __name__ == "__main__":
    # Example usage placeholder; real invocation can import these functions
    print("This script is intended to be imported and used by automation tools.")
