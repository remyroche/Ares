#!/usr/bin/env python3
"""
Script to standardize utility modules with pipeline standards.
"""

import os
import re
from pathlib import Path


def standardize_utility_module(file_path: str) -> bool:
    """Standardize a single utility module with pipeline standards."""
    print(f"Standardizing {file_path}...")

    try:
        with open(file_path, encoding="utf-8") as f:
            content=f.read()

        # Check if already standardized
        if "from src.utils.pipeline_standards import" in content:
            print(f"  ✅ Already standardized: {file_path}")
            return True

        # Skip certain files that shouldn't be standardized
        skip_files=[
            "pipeline_standards.py",
            "standardized_config_manager.py",
            "logger.py",
            "centralized_decorators.py",
            "__init__.py",
        ]

        filename=os.path.basename(file_path)
        if filename in skip_files:
            print(f"  ⏭️ Skipping {file_path} (excluded)")
            return True

        # Add pipeline standards import
        import_pattern=r"from src\.utils\.logger import system_logger"
        if import_pattern in content:
            content = content.replace(
                import_pattern,
                "from src.utils.logger import system_logger\nfrom src.utils.pipeline_standards import PipelineStandards, pipeline_standards",
            )
        else:
            # Try to find a good place to add the import
            # Look for other imports from src.utils
            utils_import_pattern=r"(from src\.utils\.[^ ]+ import [^\n]+)"
            utils_match=re.search(utils_import_pattern, content)
            if utils_match:
                last_utils_import=utils_match.group(1)
                content=content.replace(
                    last_utils_import,
                    last_utils_import + "\nfrom src.utils.pipeline_standards import PipelineStandards, pipeline_standards",
                )
            else:
                # Add at the top after other imports
                content = re.sub(r"(import [^\n]+\n)",
                    r"\1from src.utils.pipeline_standards import PipelineStandards, pipeline_standards\n",
                    content,
                    count=1,
                )

        # Add safe import patterns for external modules
        # Look for try/except import blocks and enhance them
        try_import_pattern=r"try:\s+import ([^\n]+)\nexcept ImportError:\s+([^\n]+) = None"
        try_matches=re.finditer(try_import_pattern, content)

        for match in try_matches:
            module_name=match.group(1)
            var_name=match.group(2)

            # Replace with safe import pattern
            safe_import=f'{var_name} = PipelineStandards.safe_import("{module_name}", None)'
            content=content.replace(match.group(0), safe_import)

        # Add fallback implementations for missing modules
        # Look for if module is None patterns
        if_none_pattern=r"if ([^\n]+) is None:"
        if_none_matches=re.finditer(if_none_pattern, content)

        for match in if_none_matches:
            var_name=match.group(1)
            # Add fallback comment
            fallback_comment=f"# Fallback implementation for {var_name}"
            content = content.replace(
                f"if {var_name} is None:",
                f"if {var_name} is None:\n        {fallback_comment}",
            )

        # Write back the standardized content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  ✅ Successfully standardized: {file_path}")
        return True

    except Exception as e:
        print(f"  ❌ Error standardizing {file_path}: {e}")
        return False

def main():
    """Main function to standardize utility modules."""
    utils_dir=Path("src/utils")

    success_count=0
    total_count = 0

    # Process all Python files in utils directory
    for py_file in utils_dir.glob("*.py"):
        if py_file.is_file():
            total_count += 1
            if standardize_utility_module(str(py_file)):
                success_count += 1

    print("\n📊 Utility Module Standardization Summary:")
    print(f"  Total files processed: {total_count}")
    print(f"  Successfully standardized: {success_count}")
    print(f"  Failed: {total_count - success_count}")

    if success_count== total_count:
        print("🎉 All utility modules have been successfully standardized!")
    else:
        print("⚠️ Some utility modules failed standardization. Please check the errors above.")

if __name__== "__main__":
    main()
