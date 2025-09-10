#!/usr/bin/env python3
"""
Simple Import Verifier - Standalone script to check for dead code in src/utils/
This script doesn't require heavy dependencies like numpy, pandas, etc.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import json
from datetime import datetime


class SimpleImportVerifier:
    """Simple import verifier that analyzes Python files for import relationships."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.import_relationships = {}  # file -> set of files it imports
        self.reverse_imports = {}  # file -> set of files that import it
        self.all_python_files = set()
        
    def find_python_files(self, directory: str) -> Set[str]:
        """Find all Python files in the given directory."""
        python_files = set()
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f"Warning: Directory {directory} does not exist")
            return python_files
            
        for file_path in directory_path.rglob("*.py"):
            # Skip __pycache__ directories
            if "__pycache__" in str(file_path):
                continue
            python_files.add(str(file_path))
            
        return python_files
    
    def extract_imports_from_file(self, file_path: str) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return imports
    
    def resolve_import_to_file(self, import_name: str, current_file: str) -> str:
        """Try to resolve an import name to an actual file path."""
        current_dir = Path(current_file).parent
        
        # Try relative imports first
        if import_name.startswith('.'):
            # Handle relative imports
            parts = import_name.split('.')
            if parts[0] == '':
                # Relative import from current directory
                target_path = current_dir / ('.'.join(parts[1:]) + '.py')
                if target_path.exists():
                    return str(target_path)
        else:
            # Try absolute imports
            # Look in the same directory
            target_path = current_dir / (import_name + '.py')
            if target_path.exists():
                return str(target_path)
                
            # Look in subdirectories
            for subdir in current_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    target_path = subdir / (import_name + '.py')
                    if target_path.exists():
                        return str(target_path)
                        
                    # Check for __init__.py in subdirectory
                    init_path = subdir / '__init__.py'
                    if init_path.exists():
                        # This could be the module
                        return str(init_path)
        
        return None
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """Analyze a directory for import relationships."""
        print(f"Analyzing directory: {directory}")
        
        # Find all Python files
        self.all_python_files = self.find_python_files(directory)
        print(f"Found {len(self.all_python_files)} Python files")
        
        # Initialize data structures
        for file_path in self.all_python_files:
            self.import_relationships[file_path] = set()
            self.reverse_imports[file_path] = set()
        
        # Analyze each file
        for file_path in self.all_python_files:
            print(f"Processing: {file_path}")
            imports = self.extract_imports_from_file(file_path)
            
            for import_name in imports:
                resolved_file = self.resolve_import_to_file(import_name, file_path)
                if resolved_file and resolved_file in self.all_python_files:
                    self.import_relationships[file_path].add(resolved_file)
                    self.reverse_imports[resolved_file].add(file_path)
        
        # Generate results
        results = self._generate_results()
        return results
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate analysis results."""
        import_status = {}
        
        for file_path in self.all_python_files:
            is_imported = len(self.reverse_imports[file_path]) > 0
            import_count = len(self.reverse_imports[file_path])
            
            import_status[file_path] = {
                "is_imported": is_imported,
                "import_count": import_count,
                "imported_by": list(self.reverse_imports[file_path]),
                "imports": list(self.import_relationships[file_path]),
                "module_name": Path(file_path).stem
            }
        
        # Calculate summary statistics
        total_files = len(self.all_python_files)
        imported_files = sum(1 for status in import_status.values() if status["is_imported"])
        unimported_files = total_files - imported_files
        
        # Find most and least imported files
        most_imported = max(import_status.items(), key=lambda x: x[1]["import_count"])
        least_imported = min(import_status.items(), key=lambda x: x[1]["import_count"])
        
        summary = {
            "total_files": total_files,
            "imported_files": imported_files,
            "unimported_files": unimported_files,
            "import_percentage": (imported_files / total_files * 100) if total_files > 0 else 0,
            "most_imported_file": {
                "file": most_imported[0],
                "import_count": most_imported[1]["import_count"]
            },
            "least_imported_file": {
                "file": least_imported[0],
                "import_count": least_imported[1]["import_count"]
            }
        }
        
        return {
            "import_status": import_status,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "analysis_directory": str(self.project_root)
        }
    
    def print_simple_report(self, results: Dict[str, Any]):
        """Print a simple report of the analysis."""
        summary = results["summary"]
        import_status = results["import_status"]
        
        print("\n" + "="*80)
        print("IMPORT VERIFICATION REPORT")
        print("="*80)
        print(f"Analysis Directory: {results['analysis_directory']}")
        print(f"Timestamp: {results['timestamp']}")
        print()
        print("SUMMARY:")
        print(f"  Total files analyzed: {summary['total_files']}")
        print(f"  Files imported by others: {summary['imported_files']}")
        print(f"  Files NOT imported by others: {summary['unimported_files']}")
        print(f"  Import percentage: {summary['import_percentage']:.1f}%")
        print()
        
        # Show unimported files (potential dead code)
        unimported_files = [f for f, status in import_status.items() if not status["is_imported"]]
        if unimported_files:
            print("FILES NOT IMPORTED BY OTHERS (Potential Dead Code):")
            print("-" * 60)
            for file_path in sorted(unimported_files):
                rel_path = Path(file_path).relative_to(self.project_root)
                print(f"  • {rel_path}")
            print()
        
        # Show most imported files
        print("TOP 5 MOST IMPORTED FILES:")
        print("-" * 40)
        sorted_files = sorted(import_status.items(), key=lambda x: x[1]["import_count"], reverse=True)
        for i, (file_path, status) in enumerate(sorted_files[:5], 1):
            rel_path = Path(file_path).relative_to(self.project_root)
            print(f"  {i}. {rel_path} ({status['import_count']} imports)")
            if status["imported_by"]:
                print(f"     Imported by:")
                for importer in sorted(status["imported_by"])[:3]:  # Show first 3
                    rel_importer = Path(importer).relative_to(self.project_root)
                    print(f"       • {rel_importer}")
                if len(status["imported_by"]) > 3:
                    print(f"       ... and {len(status['imported_by']) - 3} more")
            print()
        
        print("="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Import Verifier")
    parser.add_argument("--target-dir", type=str, default="src/utils", 
                       help="Target directory to analyze (default: src/utils)")
    parser.add_argument("--project-root", type=str, default=".", 
                       help="Project root directory (default: current directory)")
    parser.add_argument("--output", type=str, help="Output file for JSON results")
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = SimpleImportVerifier(args.project_root)
    
    # Run analysis
    results = verifier.analyze_directory(args.target_dir)
    
    # Print report
    verifier.print_simple_report(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()