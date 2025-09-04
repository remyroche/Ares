#!/usr/bin/env python3
"""
Simple script to run import analysis without complex dependencies.
"""

import ast
import json
import os
import sys
import importlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Any


class SimpleImportIssue:
    """Represents an import-related issue."""
    
    def __init__(self, file_path: str, line_number: int, issue_type: str,
                 message: str, severity: str = "warning", details: dict | None = None):
        self.file_path = file_path
        self.line_number = line_number
        self.issue_type = issue_type
        self.message = message
        self.severity = severity
        self.details = details or {}


class SimpleImportAnalyzer:
    """Simple import analyzer that detects unresolvable imports."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports_by_file = defaultdict(list)
        self.unresolvable_imports = []
        self.project_files = set()
        self.project_dirs = set()
        
    def _collect_project_files(self):
        """Collect all Python files in the project."""
        for root, dirs, files in os.walk(self.project_root):
            # Skip common excluded directories
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', '.pytest_cache', 'node_modules', 
                'venv', 'env', '.venv', '.env', 'build', 'dist'
            }]
            
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.project_files.add(file_path)
                    self.project_dirs.add(os.path.dirname(file_path))
    
    def analyze_directory(self) -> dict[str, Any]:
        """Analyze imports in all Python files in the project."""
        print(f"🔍 Collecting Python files in {self.project_root}...")
        self._collect_project_files()
        print(f"📁 Found {len(self.project_files)} Python files")
        
        # Analyze each file
        for file_path in self.project_files:
            try:
                self._analyze_file_imports(file_path)
            except Exception as e:
                print(f"⚠️  Error analyzing {file_path}: {e}")
        
        # Detect unresolvable imports
        self._detect_unresolvable_imports()
        
        return self._generate_report()
    
    def _analyze_file_imports(self, file_path: str) -> None:
        """Analyze imports in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        as_name = alias.asname or import_name
                        
                        file_imports.append({
                            "type": "import",
                            "module": import_name,
                            "as_name": as_name,
                            "line": node.lineno,
                        })
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        import_name = f"{module}.{alias.name}" if module else alias.name
                        as_name = alias.asname or alias.name
                        
                        file_imports.append({
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "full_name": import_name,
                            "as_name": as_name,
                            "line": node.lineno,
                        })
            
            self.imports_by_file[file_path] = file_imports
            
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
    
    def _detect_unresolvable_imports(self) -> None:
        """Detect imports that cannot be resolved."""
        print("🔍 Checking for unresolvable imports...")
        
        for file_path, imports in self.imports_by_file.items():
            file_dir = os.path.dirname(file_path)
            
            for imp in imports:
                if imp["type"] == "import":
                    module_name = imp["module"]
                    if not self._can_resolve_module(module_name, file_dir):
                        self.unresolvable_imports.append(SimpleImportIssue(
                            file_path=file_path,
                            line_number=imp["line"],
                            issue_type="unresolvable_import",
                            message=f"Cannot resolve import: {module_name}",
                            severity="error",
                            details={
                                "module": module_name,
                                "import_type": "import",
                                "reason": self._get_resolution_failure_reason(module_name, file_dir)
                            },
                        ))
                        
                elif imp["type"] == "from_import":
                    module_name = imp["module"]
                    imported_name = imp["name"]
                    
                    if not self._can_resolve_from_import(module_name, imported_name, file_dir):
                        self.unresolvable_imports.append(SimpleImportIssue(
                            file_path=file_path,
                            line_number=imp["line"],
                            issue_type="unresolvable_from_import",
                            message=f"Cannot resolve from import: {module_name}.{imported_name}",
                            severity="error",
                            details={
                                "module": module_name,
                                "name": imported_name,
                                "import_type": "from_import",
                                "reason": self._get_from_import_failure_reason(module_name, imported_name, file_dir)
                            },
                        ))
    
    def _can_resolve_module(self, module_name: str, file_dir: str) -> bool:
        """Check if a module can be resolved."""
        # Skip built-in modules and standard library
        if module_name in sys.builtin_module_names:
            return True
            
        try:
            # Try to import the module
            importlib.import_module(module_name)
            return True
        except (ImportError, SyntaxError, Exception):
            # Handle various import errors including syntax errors
            pass
        
        # Check if it's a relative import that can be resolved locally
        if module_name.startswith('.'):
            return self._can_resolve_relative_import(module_name, file_dir)
        
        # Check if it's a local module in the project
        if self._is_local_module(module_name, file_dir):
            return True
            
        return False
    
    def _can_resolve_from_import(self, module_name: str, imported_name: str, file_dir: str) -> bool:
        """Check if a from import can be resolved."""
        try:
            # Try to import the module first
            module = importlib.import_module(module_name)
            # Check if the attribute exists
            if hasattr(module, imported_name):
                return True
        except (ImportError, SyntaxError, Exception):
            # Handle various import errors including syntax errors
            pass
        
        # Check if it's a relative import
        if module_name.startswith('.'):
            return self._can_resolve_relative_from_import(module_name, imported_name, file_dir)
        
        # Check if it's a local module
        if self._is_local_module(module_name, file_dir):
            return self._can_resolve_local_from_import(module_name, imported_name, file_dir)
            
        return False
    
    def _can_resolve_relative_import(self, module_name: str, file_dir: str) -> bool:
        """Check if a relative import can be resolved."""
        # Count the number of dots to determine the level
        level = 0
        while module_name.startswith('.'):
            level += 1
            module_name = module_name[1:]
        
        if not module_name:  # Relative import without module name
            return False
            
        # Build the path based on the relative level
        current_dir = file_dir
        for _ in range(level - 1):
            current_dir = os.path.dirname(current_dir)
            if not current_dir:
                return False
        
        # Check if the module file exists
        module_file = os.path.join(current_dir, f"{module_name}.py")
        if module_file in self.project_files:
            return True
            
        # Check if it's a package
        package_dir = os.path.join(current_dir, module_name)
        init_file = os.path.join(package_dir, "__init__.py")
        if init_file in self.project_files:
            return True
            
        return False
    
    def _can_resolve_relative_from_import(self, module_name: str, imported_name: str, file_dir: str) -> bool:
        """Check if a relative from import can be resolved."""
        # First check if the module can be resolved
        if not self._can_resolve_relative_import(module_name, file_dir):
            return False
            
        # For now, assume the attribute exists if the module can be resolved
        return True
    
    def _is_local_module(self, module_name: str, file_dir: str) -> bool:
        """Check if a module is a local module in the project."""
        # Check if there's a corresponding .py file
        module_file = os.path.join(file_dir, f"{module_name}.py")
        if module_file in self.project_files:
            return True
            
        # Check if it's a package
        package_dir = os.path.join(file_dir, module_name)
        init_file = os.path.join(package_dir, "__init__.py")
        if init_file in self.project_files:
            return True
            
        # Check parent directories
        current_dir = file_dir
        while current_dir:
            module_file = os.path.join(current_dir, f"{module_name}.py")
            if module_file in self.project_files:
                return True
                
            package_dir = os.path.join(current_dir, module_name)
            init_file = os.path.join(package_dir, "__init__.py")
            if init_file in self.project_files:
                return True
                
            current_dir = os.path.dirname(current_dir)
            if current_dir == os.path.dirname(current_dir):  # Reached root
                break
                
        return False
    
    def _can_resolve_local_from_import(self, module_name: str, imported_name: str, file_dir: str) -> bool:
        """Check if a local from import can be resolved."""
        # For now, assume the attribute exists if the module is local
        return True
    
    def _get_resolution_failure_reason(self, module_name: str, file_dir: str) -> str:
        """Get a human-readable reason why a module cannot be resolved."""
        if module_name.startswith('.'):
            return "Relative import path cannot be resolved"
        elif self._is_local_module(module_name, file_dir):
            return "Local module exists but may have import issues"
        else:
            return "Module not found in Python path or project"
    
    def _get_from_import_failure_reason(self, module_name: str, imported_name: str, file_dir: str) -> str:
        """Get a human-readable reason why a from import cannot be resolved."""
        if module_name.startswith('.'):
            return "Relative import path cannot be resolved"
        elif self._is_local_module(module_name, file_dir):
            return f"Module exists but '{imported_name}' attribute not found"
        else:
            return "Module not found in Python path or project"
    
    def _generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive import analysis report."""
        return {
            "summary": {
                "total_files_analyzed": len(self.imports_by_file),
                "total_imports": sum(len(imports) for imports in self.imports_by_file.values()),
                "total_issues": len(self.unresolvable_imports),
                "unresolvable_imports": len(self.unresolvable_imports),
            },
            "issues": {
                "unresolvable_imports": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details,
                    }
                    for issue in self.unresolvable_imports
                ],
            },
        }


def main():
    """Run the simple import analysis."""
    print("🔍 Running Simple Import Analysis...")
    print("=" * 60)
    
    # Get the project root (parent of code_quality directory)
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Initialize the analyzer
    analyzer = SimpleImportAnalyzer(str(project_root))
    
    # Analyze the entire project
    results = analyzer.analyze_directory()
    
    # Print summary
    summary = results["summary"]
    print(f"\n📊 Analysis Summary:")
    print(f"  Files analyzed: {summary['total_files_analyzed']}")
    print(f"  Total imports: {summary['total_imports']}")
    print(f"  Total issues: {summary['total_issues']}")
    print(f"  Unresolvable imports: {summary['unresolvable_imports']}")
    
    # Show unresolvable imports in detail
    unresolvable = results["issues"]["unresolvable_imports"]
    if unresolvable:
        print(f"\n❌ Unresolvable Imports ({len(unresolvable)} found):")
        print("-" * 60)
        for issue in unresolvable:
            print(f"  📄 {issue['file']}:{issue['line']}")
            print(f"     {issue['message']}")
            if 'reason' in issue['details']:
                print(f"     Reason: {issue['details']['reason']}")
            print()
    else:
        print(f"\n✅ No unresolvable imports found!")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/simple_import_analysis_{timestamp}.json"
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")
    
    # Return exit code based on critical issues
    if summary['unresolvable_imports'] > 0:
        print(f"\n⚠️  Found {summary['unresolvable_imports']} unresolvable import issues that need attention!")
        return 1
    else:
        print(f"\n✅ No unresolvable import issues found!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
