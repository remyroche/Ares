#!/usr/bin/env python3
"""
Verify the test structure and provide information about running the tests.
"""

import ast
import sys
from pathlib import Path


def analyze_test_file():
    """Analyze the test file structure without running it."""
    test_file = Path(__file__).parent / "tests" / "test_common_operations.py"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1

    print(f"✅ Test file found: {test_file}")
    print(f"   Size: {test_file.stat().st_size:,} bytes")

    # Parse the test file to extract test classes and methods
    try:
        with open(test_file) as f:
            tree = ast.parse(f.read())

        test_classes = []
        test_methods = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                test_classes.append(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        test_methods.append(f"{node.name}.{item.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        print("\n📋 Test Structure Analysis:")
        print(f"   Test Classes: {len(test_classes)}")
        print(f"   Test Methods: {len(test_methods)}")

        print("\n📦 Test Classes Found:")
        for cls in test_classes:
            print(f"   - {cls}")

        print("\n🧪 Sample Test Methods (first 10):")
        for method in test_methods[:10]:
            print(f"   - {method}")
        if len(test_methods) > 10:
            print(f"   ... and {len(test_methods) - 10} more")

        # Check for required imports
        required_deps = ["numpy", "pandas"]
        missing_deps = []
        for dep in required_deps:
            if not any(dep in imp for imp in imports):
                missing_deps.append(dep)

        print("\n📚 Dependencies:")
        print(f"   Required: {', '.join(required_deps)}")
        if missing_deps:
            print(f"   ⚠️  Missing from imports: {', '.join(missing_deps)}")
        else:
            print("   ✅ All required dependencies are imported")

        print("\n🔧 To run these tests properly:")
        print("   1. Create a virtual environment:")
        print("      python3 -m venv venv")
        print("      source venv/bin/activate  # On Linux/Mac")
        print("   2. Install dependencies:")
        print("      pip install numpy pandas")
        print("   3. Run the tests:")
        print("      python run_common_operations_tests.py")
        print("      # or with coverage:")
        print("      python run_common_operations_tests.py --coverage")

        print("\n📝 Test Module Target:")
        print("   The tests are for: src/utils/common_operations.py")

        return 0

    except Exception as e:
        print(f"\n❌ Error analyzing test file: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(analyze_test_file())
