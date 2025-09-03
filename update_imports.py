#!/usr/bin/env python3
"""
Script to update imports in all analyzer files to use minimal modules.
"""

from pathlib import Path


def update_file_imports(file_path: str):
    """Update imports in a single file."""
    with open(file_path) as f:
        content = f.read()

    # Update the problematic imports
    old_imports = [
        "from ..core.config import CodeQualityConfig, get_default_config",
        "from ..utils.file_utils import find_python_files",
    ]

    new_imports = [
        "import sys\nsys.path.insert(0, str(Path(__file__).parent.parent.parent))\nfrom minimal_config import CodeQualityConfig, get_default_config",
        "from minimal_file_utils import find_python_files",
    ]

    for old_import, new_import in zip(old_imports, new_imports, strict=False):
        if old_import in content:
            content = content.replace(old_import, new_import)

    # Write back to file
    with open(file_path, "w") as f:
        f.write(content)

    print(f"Updated: {file_path}")

def main():
    """Update all analyzer files."""
    analyzers_dir = Path("code_quality/analyzers")

    for analyzer_file in analyzers_dir.glob("*.py"):
        if analyzer_file.name != "__init__.py":
            update_file_imports(str(analyzer_file))

if __name__ == "__main__":
    main()
