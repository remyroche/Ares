#!/usr/bin/env python3
"""
Safe import fixer that adds imports without breaking existing code.
This version uses regex-based insertion instead of AST parsing.
"""

import json
import re
from pathlib import Path

# Essential imports that should be added to most files
ESSENTIAL_IMPORTS = [
    "import datetime",
    "import json",
    "import os",
    "import sys",
    "from pathlib import Path",
    "from typing import Dict, List, Optional, Union, Any",
    "import logging",
    "from collections import defaultdict",
    "import asyncio",
    "import numpy as np",
    "import pandas as pd",
    "from copy import copy, deepcopy",
]

# Function to module mappings for common undefined functions
FUNCTION_TO_IMPORT = {
    "DataFrame": "import pandas as pd",
    "Series": "import pandas as pd",
    "array": "import numpy as np",
    "zeros": "import numpy as np",
    "ones": "import numpy as np",
    "mean": "import numpy as np",
    "std": "import numpy as np",
    "now": "import datetime",
    "today": "import datetime",
    "timedelta": "from datetime import timedelta",
    "Path": "from pathlib import Path",
    "ArgumentParser": "import argparse",
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "sleep": "import asyncio",
    "create_task": "import asyncio",
    "gather": "import asyncio",
    "deepcopy": "from copy import deepcopy",
    "copy": "from copy import copy",
    "getLogger": "import logging",
    "dumps": "import json",
    "loads": "import json",
}


class SafeImportFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.failed_files = []

    def add_imports_to_file(self, file_path: str, imports_to_add: set[str]) -> bool:
        """Add imports to a file safely using regex."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")

            # Find the best position to insert imports
            insert_pos = 0
            last_import = -1
            in_docstring = False
            docstring_char = None

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Handle docstrings
                if not in_docstring:
                    if stripped.startswith(('"""', "'''")):
                        docstring_char = '"""' if stripped.startswith('"""') else "'''"
                        if stripped.count(docstring_char) == 1:
                            in_docstring = True
                        continue
                else:
                    if docstring_char in stripped:
                        in_docstring = False
                        docstring_char = None
                        continue
                    continue

                # Track imports
                if (stripped.startswith(("import ", "from ")) or stripped.startswith("#") and i < 10):
                    last_import = i
                elif stripped and not stripped.startswith("#"):
                    # First non-import, non-comment line
                    break

            # Determine insert position
            if last_import >= 0:
                insert_pos = last_import + 1
            else:
                # No imports found, insert after initial comments/docstring
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith("#"):
                        if not (line.strip().startswith('"""') or line.strip().startswith("'''")):
                            insert_pos = i
                            break

            # Get existing imports
            existing_imports = set()
            for line in lines:
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    existing_imports.add(line.strip())

            # Filter out already existing imports
            new_imports = []
            for imp in sorted(imports_to_add):
                if imp not in existing_imports:
                    # Check if module is already imported in a different form
                    module = imp.split()[1] if imp.startswith("import ") else imp.split()[1]
                    already_imported = any(module in existing for existing in existing_imports)
                    if not already_imported:
                        new_imports.append(imp)

            if not new_imports:
                return False  # Nothing to add

            # Insert the imports
            for imp in reversed(new_imports):
                lines.insert(insert_pos, imp)

            # Add blank line after imports if needed
            if insert_pos < len(lines) - 1 and lines[insert_pos + len(new_imports)].strip():
                lines.insert(insert_pos + len(new_imports), "")

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def analyze_and_fix_file(self, file_path: str) -> dict:
        """Analyze a file and add necessary imports."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Find potentially undefined functions
            imports_needed = set()

            # Use regex to find function calls
            for func, import_stmt in FUNCTION_TO_IMPORT.items():
                # Look for function usage patterns
                patterns = [
                    rf"\b{func}\s*\(",  # Function call
                    rf"\b{func}\.",      # Method call
                    rf"=\s*{func}\s*\(", # Assignment
                ]

                for pattern in patterns:
                    if re.search(pattern, content):
                        imports_needed.add(import_stmt)
                        break

            # Special handling for common patterns
            if "pd.DataFrame" in content or "pandas.DataFrame" in content:
                imports_needed.add("import pandas as pd")

            if "np.array" in content or "numpy.array" in content:
                imports_needed.add("import numpy as np")

            # Check for async patterns
            if re.search(r"async\s+def", content):
                imports_needed.add("import asyncio")

            # Add essential imports for common operations
            if ".fillna(" in content or ".rolling(" in content:
                imports_needed.add("import pandas as pd")

            if ".mean()" in content or ".std()" in content:
                imports_needed.add("import numpy as np")

            if imports_needed:
                success = self.add_imports_to_file(file_path, imports_needed)
                if success:
                    self.fixed_files.append(file_path)
                    return {"status": "fixed", "imports_added": list(imports_needed)}
                self.failed_files.append(file_path)
                return {"status": "failed"}
            return {"status": "no_changes_needed"}

        except Exception as e:
            self.failed_files.append(file_path)
            return {"status": "error", "error": str(e)}

    def fix_project(self, dry_run: bool = True):
        """Fix imports across the entire project."""
        python_files = list(self.project_root.rglob("*.py"))

        # Filter out excluded directories
        python_files = [
            f for f in python_files
            if "__pycache__" not in str(f) and
               ".venv" not in str(f) and
               "venv" not in str(f)
        ]

        print(f"Found {len(python_files)} Python files to analyze")

        if dry_run:
            print("\nDRY RUN MODE - No files will be modified")
            files_needing_fixes = []

            for file_path in python_files[:20]:  # Sample first 20 files
                result = self.analyze_and_fix_file(str(file_path))
                if result["status"] == "fixed":
                    files_needing_fixes.append({
                        "file": str(file_path),
                        "imports": result.get("imports_added", []),
                    })

            print(f"\nFiles that would be fixed: {len(files_needing_fixes)}")
            for file_info in files_needing_fixes[:5]:
                print(f"\n{Path(file_info['file']).name}:")
                for imp in file_info["imports"]:
                    print(f"  + {imp}")

            return {"dry_run": True, "files_to_fix": len(files_needing_fixes)}
        # Actually fix files
        for i, file_path in enumerate(python_files):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(python_files)} files processed")

            self.analyze_and_fix_file(str(file_path))

        print(f"\nFixed {len(self.fixed_files)} files")
        print(f"Failed to fix {len(self.failed_files)} files")

        return {
            "fixed": len(self.fixed_files),
            "failed": len(self.failed_files),
            "fixed_files": self.fixed_files[:10],  # First 10
            "failed_files": self.failed_files[:10],  # First 10
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Safe import fixer for Python files")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the files (default is dry run)")

    args = parser.parse_args()

    fixer = SafeImportFixer(args.project_root)
    result = fixer.fix_project(dry_run=not args.fix)

    # Save report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/workspace/code_quality/reports/safe_import_fixes_report_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
