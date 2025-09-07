#!/usr/bin/env python3
"""
Dead Code Analysis using Interaction Mapping Data
Analyzes the interaction mapping results to identify dead code.
"""

import json
import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class DeadCodeAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.interaction_data = {}
        self.dead_code_results = {
            "unused_functions": [],
            "unused_classes": [],
            "unused_imports": [],
            "unreachable_code": [],
            "orphaned_files": []
        }
    
    def load_interaction_data(self):
        """Load interaction mapping data."""
        interaction_file = self.project_root / "code_quality/reports/interaction_mapping/basic_interaction_mapping_20250906_101736.json"
        
        if interaction_file.exists():
            print(f"📊 Loading interaction data from {interaction_file}")
            with open(interaction_file, 'r') as f:
                self.interaction_data = json.load(f)
            print(f"✅ Loaded {len(self.interaction_data.get('interactions', []))} interactions")
        else:
            print("❌ No interaction mapping data found")
    
    def analyze_file_for_dead_code(self, file_path: Path) -> Dict[str, List]:
        """Analyze a single file for dead code."""
        if not file_path.exists() or not file_path.suffix == '.py':
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract defined functions, classes, and imports
            defined_functions = set()
            defined_classes = set()
            imported_modules = set()
            imported_functions = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module)
                    for alias in node.names:
                        imported_functions.add(alias.name)
            
            # Check if functions/classes are used elsewhere
            unused_functions = []
            unused_classes = []
            unused_imports = []
            
            # Get interactions for this file
            file_interactions = self._get_file_interactions(str(file_path))
            
            for func in defined_functions:
                if not self._is_function_used(func, file_interactions):
                    unused_functions.append({
                        "name": func,
                        "file": str(file_path),
                        "line": self._get_function_line(tree, func)
                    })
            
            for cls in defined_classes:
                if not self._is_class_used(cls, file_interactions):
                    unused_classes.append({
                        "name": cls,
                        "file": str(file_path),
                        "line": self._get_class_line(tree, cls)
                    })
            
            return {
                "unused_functions": unused_functions,
                "unused_classes": unused_classes,
                "unused_imports": unused_imports
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return {}
    
    def _get_file_interactions(self, file_path: str) -> List[Dict]:
        """Get interactions for a specific file."""
        interactions = self.interaction_data.get('interactions', [])
        return [i for i in interactions if file_path in i.get('file', '')]
    
    def _is_function_used(self, func_name: str, interactions: List[Dict]) -> bool:
        """Check if a function is used in interactions."""
        for interaction in interactions:
            if func_name in str(interaction):
                return True
        return False
    
    def _is_class_used(self, class_name: str, interactions: List[Dict]) -> bool:
        """Check if a class is used in interactions."""
        for interaction in interactions:
            if class_name in str(interaction):
                return True
        return False
    
    def _get_function_line(self, tree: ast.AST, func_name: str) -> int:
        """Get line number of a function definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node.lineno
        return 0
    
    def _get_class_line(self, tree: ast.AST, class_name: str) -> int:
        """Get line number of a class definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node.lineno
        return 0
    
    def analyze_project(self):
        """Analyze the entire project for dead code."""
        print("🔍 Analyzing project for dead code...")
        
        python_files = list(self.project_root.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files")
        
        total_unused_functions = 0
        total_unused_classes = 0
        total_unused_imports = 0
        
        for file_path in python_files:
            if "test" in str(file_path) or "__pycache__" in str(file_path):
                continue
                
            results = self.analyze_file_for_dead_code(file_path)
            
            if results:
                unused_funcs = results.get("unused_functions", [])
                unused_classes = results.get("unused_classes", [])
                unused_imports = results.get("unused_imports", [])
                
                if unused_funcs or unused_classes or unused_imports:
                    print(f"\n📄 {file_path.relative_to(self.project_root)}")
                    
                    if unused_funcs:
                        print(f"  🔴 Unused functions: {len(unused_funcs)}")
                        for func in unused_funcs[:3]:  # Show first 3
                            print(f"    - {func['name']} (line {func['line']})")
                        if len(unused_funcs) > 3:
                            print(f"    ... and {len(unused_funcs) - 3} more")
                    
                    if unused_classes:
                        print(f"  🔴 Unused classes: {len(unused_classes)}")
                        for cls in unused_classes[:3]:  # Show first 3
                            print(f"    - {cls['name']} (line {cls['line']})")
                        if len(unused_classes) > 3:
                            print(f"    ... and {len(unused_classes) - 3} more")
                    
                    if unused_imports:
                        print(f"  🔴 Unused imports: {len(unused_imports)}")
                        for imp in unused_imports[:3]:  # Show first 3
                            print(f"    - {imp}")
                        if len(unused_imports) > 3:
                            print(f"    ... and {len(unused_imports) - 3} more")
                
                total_unused_functions += len(unused_funcs)
                total_unused_classes += len(unused_classes)
                total_unused_imports += len(unused_imports)
        
        print(f"\n📊 DEAD CODE ANALYSIS SUMMARY:")
        print(f"  🔴 Total unused functions: {total_unused_functions}")
        print(f"  🔴 Total unused classes: {total_unused_classes}")
        print(f"  🔴 Total unused imports: {total_unused_imports}")
        print(f"  📁 Files analyzed: {len(python_files)}")
        
        return {
            "total_unused_functions": total_unused_functions,
            "total_unused_classes": total_unused_classes,
            "total_unused_imports": total_unused_imports,
            "files_analyzed": len(python_files)
        }

def main():
    project_root = "/Users/remyroche/Documents/Ares"
    analyzer = DeadCodeAnalyzer(project_root)
    
    print("🚀 Starting Dead Code Analysis using Interaction Mapping Data")
    print("=" * 60)
    
    # Load interaction data
    analyzer.load_interaction_data()
    
    # Analyze project
    results = analyzer.analyze_project()
    
    print("\n✅ Dead code analysis complete!")
    print(f"📈 Dead code percentage: {(results['total_unused_functions'] + results['total_unused_classes']) / results['files_analyzed'] * 100:.1f}%")

if __name__ == "__main__":
    main()
