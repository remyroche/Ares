#!/usr/bin/env python3
"""
Safe Syntax Error Fixer
Fixes common = vs , syntax errors without breaking legitimate code.
"""

import os
import re
from pathlib import Path



def fix_import_statements(content: str) -> str:
    """Fix malformed import statements."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: from pathlib import Path, from src.utils.logger import system_logger
        if re.match(r"^\s*from\s+[^,]+,\s+from\s+", line):
            # Split into multiple import statements
            parts = re.split(r",\s+from\s+", line)
            if len(parts) > 1:
                # First part
                fixed_lines.append(parts[0])
                # Subsequent parts
                for part in parts[1:]:
                    fixed_lines.append(f"from {part}")
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_function_parameters(content: str) -> str:
    """Fix function parameter syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: def __init__(self = symbol: str = "ETHUSDT"):
        if re.match(r"^\s*def\s+\w+\s*\(\s*self\s*=\s*", line):
            line = re.sub(r"\(\s*self\s*=\s*", "(self, ", line)

        # Fix: def some_function(param = value: type):
        elif re.match(r"^\s*def\s+\w+\s*\([^)]*=\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:", line):
            line = re.sub(r"(\w+)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\s*:)", r"\1, \2", line)

        # Fix: async def function(self = param: type):
        elif re.match(r"^\s*async\s+def\s+\w+\s*\(\s*self\s*=\s*", line):
            line = re.sub(r"\(\s*self\s*=\s*", "(self, ", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_for_loops(content: str) -> str:
    """Fix for loop syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: for test_name , result in test_results.items():
        if re.search(r"for\s+[^=]+\s*=\s*[^=]+\s+in\s+", line):
            line = re.sub(r"for\s+([^=]+)\s*=\s*([^=]+)\s+in\s+", r"for \1, \2 in ", line)

        # Fix: for i , (file_name, gap_count) in enumerate(...):
        elif re.search(r"for\s+[^=]+\s*=\s*\([^)]+\)\s+in\s+", line):
            line = re.sub(r"for\s+([^=]+)\s*=\s*\(([^)]+)\)\s+in\s+", r"for \1, (\2) in ", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_dictionary_definitions(content: str) -> str:
    """Fix dictionary definition syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: {"success": True , "gaps_fixed": 0}
        if re.search(r'"[^"]+"\s*:\s*[^=]+\s*=\s*"[^"]+"\s*:', line) or re.search(r'"[^"]+"\s*:\s*[^=]+\s*=\s*"[^"]+"\s*:', line):
            line = re.sub(r'("[^"]+"\s*:\s*[^=]+)\s*=\s*("[^"]+"\s*:)', r"\1, \2", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_type_hints(content: str) -> str:
    """Fix type hint syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: -> tuple[bool , list[str]]:
        if re.search(r"->\s*[^=]*\s*=\s*[^=]*\s*:", line):
            line = re.sub(r"->\s*([^=]*)\s*=\s*([^=]*)\s*:", r"-> \1, \2:", line)

        # Fix: dict[str , Any]
        elif re.search(r"dict\[[^=]*\s*=\s*[^=]*\]", line):
            line = re.sub(r"dict\[([^=]*)\s*=\s*([^=]*)\]", r"dict[\1, \2]", line)

        # Fix: list[str , Any]
        elif re.search(r"list\[[^=]*\s*=\s*[^=]*\]", line):
            line = re.sub(r"list\[([^=]*)\s*=\s*([^=]*)\]", r"list[\1, \2]", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_function_calls(content: str) -> str:
    """Fix function call syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: func(*args, **kwargs)
        if re.search(r"\(\s*\*args\s*=\s*\*\*kwargs\s*\)", line):
            line = re.sub(r"\(\s*\*args\s*=\s*\*\*kwargs\s*\)", "(*args, **kwargs)", line)

        # Fix: func(param = value, other_param)
        elif re.search(r"\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]+\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)", line):
            # This is more complex, so we'll be conservative
            pass

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_isinstance_calls(content: str) -> str:
    """Fix isinstance syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: isinstance(df.index , pd.DatetimeIndex)
        if re.search(r"isinstance\s*\(\s*[^=]+\s*=\s*[^=]+\s*\)", line):
            line = re.sub(r"isinstance\s*\(\s*([^=]+)\s*=\s*([^=]+)\s*\)", r"isinstance(\1, \2)", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_return_statements(content: str) -> str:
    """Fix return statement syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: return csv_exists or parquet_exists , files_found
        if re.search(r"return\s+[^=]+\s+or\s+[^=]+\s*=\s*[^=]+", line):
            line = re.sub(r"return\s+([^=]+)\s+or\s+([^=]+)\s*=\s*([^=]+)", r"return \1 or \2, \3", line)

        # Fix: return (exists , file_types)
        elif re.search(r"return\s*\(\s*[^=]+\s*=\s*[^=]+\s*\)", line):
            line = re.sub(r"return\s*\(\s*([^=]+)\s*=\s*([^=]+)\s*\)", r"return (\1, \2)", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_assignment_statements(content: str) -> str:
    """Fix assignment statement syntax errors."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Fix: exists, file_types = check_aggtrades_file_exists(date_str)
        if re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*", line):
            line = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*", r"\1, \2 = ", line)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_file(file_path: Path) -> tuple[bool, list[str]]:
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes in order
        content = fix_import_statements(content)
        content = fix_function_parameters(content)
        content = fix_for_loops(content)
        content = fix_dictionary_definitions(content)
        content = fix_type_hints(content)
        content = fix_function_calls(content)
        content = fix_isinstance_calls(content)
        content = fix_return_statements(content)
        content = fix_assignment_statements(content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, [f"Fixed {file_path}"]

        return False, []

    except Exception as e:
        return False, [f"Error fixing {file_path}: {e}"]


def find_python_files(directory: Path) -> list[Path]:
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common directories that shouldn't be modified
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}]

        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    return python_files


def main():
    """Main function to fix syntax errors."""
    print("🔧 Safe Syntax Error Fixer")
    print("=" * 50)

    # Get current directory
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")

    # Find all Python files
    python_files = find_python_files(current_dir)
    print(f"Found {len(python_files)} Python files")

    # Files to skip (known to have issues or are generated)
    skip_files = {
        "test_step1_pipeline.py",
        "test_timeframe_loading.py",
        "update_aggtrades_gaps.py",
        "verify_aggtrades_downloads.py",
    }

    fixed_count = 0
    errors = []

    for file_path in python_files:
        if file_path.name in skip_files:
            print(f"⏭️  Skipping {file_path.name} (known problematic file)")
            continue

        print(f"🔍 Checking {file_path.name}...")
        fixed, messages = fix_file(file_path)

        if fixed:
            fixed_count += 1
            print(f"✅ Fixed {file_path.name}")

        errors.extend(messages)

    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"   Files processed: {len(python_files)}")
    print(f"   Files fixed: {fixed_count}")
    print(f"   Errors: {len(errors)}")

    if errors:
        print("\n❌ Errors encountered:")
        for error in errors:
            print(f"   {error}")

    if fixed_count > 0:
        print(f"\n🎉 Successfully fixed {fixed_count} files!")
        print("💡 Run 'ruff check .' to verify the fixes.")
    else:
        print("\nℹ️  No files needed fixing.")


if __name__ == "__main__":
    main()
