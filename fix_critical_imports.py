#!/usr/bin/env python3
"""
Fix critical import issues in src/utils files.
"""

import os
import re


def fix_centralized_decorators():
    """Fix missing imports in centralized_decorators.py"""
    file_path = "src/utils/centralized_decorators.py"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path) as f:
        content = f.read()

    # Add missing imports at the top
    import_section = """import functools
import logging
from typing import A, Callableny, Callable, Dict, Optional, List, Union, TypeVar, cast
import time
import warnings
from datetime import datetime
import inspect
import json

# Handle optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
"""

    # Check if imports already exist
    if "PANDAS_AVAILABLE" not in content:
        # Find the right place to insert (after existing imports)
        import_pattern = r"^(import.*\n|from.*import.*\n)+"
        match = re.search(import_pattern, content, re.MULTILINE)

        if match:
            insert_pos = match.end()
            # Insert the new imports after existing imports
            content = content[:insert_pos] + "\n" + import_section.split("\n\n")[-1] + "\n" + content[insert_pos:]
        else:
            # Add at the beginning after docstring
            lines = content.split("\n")
            insert_idx = 0

            # Skip docstring if present
            if lines[0].startswith('"""'):
                for i, line in enumerate(lines[1:], 1):
                    if line.strip().endswith('"""'):
                        insert_idx = i + 1
                        break

            lines.insert(insert_idx, import_section)
            content = "\n".join(lines)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"Fixed imports in {file_path}")

def fix_observability_imports():
    """Fix sentry_sdk import in observability.py"""
    file_path = "src/utils/observability.py"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path) as f:
        content = f.read()

    # Replace the problematic import with try-except
    old_import = "import sentry_sdk"
    new_import = """try:
    import sentry_sdk
    from sentry_sdk.integrations.aiohttp import AioHttpIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    sentry_sdk = None
    AioHttpIntegration = None
    SENTRY_AVAILABLE = False"""

    if old_import in content and "SENTRY_AVAILABLE" not in content:
        # Find all sentry imports and replace them
        content = re.sub(
            r"(import sentry_sdk.*\n)(.*from sentry_sdk.*\n)*",
            new_import + "\n",
            content,
        )

        # Update code that uses sentry_sdk to check availability
        content = re.sub(
            r"(\s+)(sentry_sdk\.init\()",
            r"\1if SENTRY_AVAILABLE and sentry_sdk:\n\1    \2",
            content,
        )

    with open(file_path, "w") as f:
        f.write(content)

    print(f"Fixed imports in {file_path}")

def add_missing_imports_to_file(file_path, undefined_names):
    """Add missing imports based on undefined names."""
    import_map = {
        "Path": "from pathlib import Path",
        "List": "from typing import List",
        "Dict": "from typing import Dict",
        "Optional": "from typing import Optional",
        "Any": "from typing import A, Callableny",
        "Union": "from typing import Union",
        "Tuple": "from typing import Tuple",
        "Set": "from typing import Set",
        "Type": "from typing import Type",
        "Callable": "from typing import Callable",
        "TypeVar": "from typing import TypeVar",
        "datetime": "from datetime import datetime",
        "timedelta": "from datetime import timedelta",
        "defaultdict": "from collections import defaultdict",
        "Counter": "from collections import Counter",
        "json": "import json",
        "os": "import os",
        "sys": "import sys",
        "re": "import re",
        "time": "import time",
        "logging": "import logging",
        "warnings": "import warnings",
        "inspect": "import inspect",
        "functools": "import functools",
        "asyncio": "import asyncio",
        "abc": "import abc",
        "ABC": "from abc import ABC",
        "abstractmethod": "from abc import abstractmethod",
    }

    if not os.path.exists(file_path):
        return

    with open(file_path) as f:
        content = f.read()

    # Collect imports to add
    imports_to_add = []
    for name in undefined_names:
        if name in import_map and import_map[name] not in content:
            imports_to_add.append(import_map[name])

    if imports_to_add:
        # Find where to insert imports
        lines = content.split("\n")
        insert_idx = 0

        # Skip shebang and encoding
        for i, line in enumerate(lines):
            if not line.startswith("#") and not line.startswith('"""'):
                insert_idx = i
                break

        # Skip docstring if present
        if lines[insert_idx].startswith('"""'):
            for i, line in enumerate(lines[insert_idx+1:], insert_idx+1):
                if line.strip().endswith('"""'):
                    insert_idx = i + 1
                    break

        # Add imports
        for imp in sorted(imports_to_add):
            lines.insert(insert_idx, imp)
            insert_idx += 1

        content = "\n".join(lines)

        with open(file_path, "w") as f:
            f.write(content)

        print(f"Added {len(imports_to_add)} imports to {file_path}")

def main():
    """Run all import fixes."""
    print("Fixing critical import issues...")
    print("=" * 60)

    # Fix specific known issues
    fix_centralized_decorators()
    fix_observability_imports()

    # Scan for other undefined names
    print("\nScanning for other undefined names...")

    # Run flake8 to find undefined names
    import subprocess
    result = subprocess.run(
        ["flake8", "src/utils", "--select=F821", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"],
        check=False, capture_output=True,
        text=True,
    )

    if result.stdout:
        # Parse undefined names by file
        undefined_by_file = {}
        for line in result.stdout.strip().split("\n"):
            if "undefined name" in line:
                parts = line.split(":")
                if len(parts) >= 4:
                    file_path = parts[0]
                    match = re.search(r"undefined name '(\w+)'", line)
                    if match:
                        name = match.group(1)
                        if file_path not in undefined_by_file:
                            undefined_by_file[file_path] = set()
                        undefined_by_file[file_path].add(name)

        # Fix undefined names
        for file_path, names in undefined_by_file.items():
            print(f"\nFixing undefined names in {file_path}: {names}")
            add_missing_imports_to_file(file_path, names)

    print("\nImport fixes completed!")

if __name__ == "__main__":
    main()
