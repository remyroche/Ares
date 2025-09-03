#!/usr/bin/env python3
"""
Fix common syntax patterns that are causing errors across the codebase.
"""

from pathlib import Path


def fix_arrow_syntax(content: str) -> str:
    """Fix incorrect :-> syntax to ->"""
    return content.replace(":-> ", "-> ")


def fix_common_patterns(file_path: Path) -> bool:
    """Fix common syntax patterns in a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_arrow_syntax(content)

        # Add more pattern fixes here as we discover them

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix common syntax patterns across the codebase."""
    project_root = Path("/workspace/src")
    python_files = list(project_root.rglob("*.py"))

    fixed_count = 0

    print(f"Checking {len(python_files)} Python files for common syntax patterns...")

    for file_path in python_files:
        if fix_common_patterns(file_path):
            fixed_count += 1
            print(f"✓ Fixed {file_path.name}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
