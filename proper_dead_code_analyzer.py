#!/usr/bin/env python3
"""
Proper Dead Code Analysis using Interaction Mapping Data
Correctly analyzes the interaction mapping results to identify dead code.
"""

import json
import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class ProperDeadCodeAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.interaction_data = {}
        self.used_functions = set()
        self.used_classes = set()
        self.used_methods = set()
        
    def load_interaction_data(self):
        """Load interaction mapping data."""
        interaction_file = self.project_root / "code_quality/reports/interaction_mapping/basic_interaction_mapping_20250906_101736.json"
        
        if interaction_file.exists():
            print(f"📊 Loading interaction data from {interaction_file}")
            with open(interaction_file, 'r') as f:
                self.interaction_data = json.load(f)
            
            # Extract all used functions and classes from interactions
            self._extract_used_entities()
            print(f"✅ Loaded {len(self.interaction_data.get('results', {}).get('interactions', []))} interactions")
            print(f"📈 Found {len(self.used_functions)} used functions")
            print(f"📈 Found {len(self.used_classes)} used classes")
        else:
            print("❌ No interaction mapping data found")
    
    def _extract_used_entities(self):
        """Extract all used functions and classes from interaction data."""
        interactions = self.interaction_data.get('results', {}).get('interactions', [])
        
        for interaction in interactions:
            if interaction.get('type') == 'function_call':
                # Add the target function as used
                target = interaction.get('target', '')
                if target:
                    self.used_functions.add(target)
                    
                    # If it's a method call (contains '.'), also add the class
                    if '.' in target:
                        class_name = target.split('.')[0]
                        self.used_classes.add(class_name)
                        self.used_methods.add(target)
            
            elif interaction.get('type') == 'class_instantiation':
                class_name = interaction.get('target', '')
                if class_name:
                    self.used_classes.add(class_name)
            
            elif interaction.get('type') == 'import':
                # Imported modules/classes are considered used
                target = interaction.get('target', '')
                if target:
                    if '.' in target:
                        # It's a class import
                        class_name = target.split('.')[-1]
                        self.used_classes.add(class_name)
                    else:
                        # It's a module import
                        self.used_functions.add(target)
    
    def analyze_file_for_dead_code(self, file_path: Path) -> Dict[str, List]:
        """Analyze a single file for dead code."""
        if not file_path.exists() or not file_path.suffix == '.py':
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract defined functions, classes, and methods
            defined_functions = set()
            defined_classes = set()
            defined_methods = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_functions.add(node.name)
                    # If it's a method (inside a class), add it to methods
                    if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if hasattr(parent, 'body') and node in parent.body):
                        # Find the class name
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef) and hasattr(parent, 'body') and node in parent.body:
                                method_name = f"{parent.name}.{node.name}"
                                defined_methods.add(method_name)
                                break
                elif isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)
            
            # Check what's actually used
            unused_functions = []
            unused_classes = []
            unused_methods = []
            
            # Check functions
            for func in defined_functions:
                if func not in self.used_functions and not self._is_special_function(func):
                    unused_functions.append({
                        "name": func,
                        "file": str(file_path),
                        "line": self._get_function_line(tree, func)
                    })
            
            # Check classes
            for cls in defined_classes:
                if cls not in self.used_classes and not self._is_special_class(cls):
                    unused_classes.append({
                        "name": cls,
                        "file": str(file_path),
                        "line": self._get_class_line(tree, cls)
                    })
            
            # Check methods
            for method in defined_methods:
                if method not in self.used_methods and not self._is_special_method(method):
                    unused_methods.append({
                        "name": method,
                        "file": str(file_path),
                        "line": self._get_method_line(tree, method)
                    })
            
            return {
                "unused_functions": unused_functions,
                "unused_classes": unused_classes,
                "unused_methods": unused_methods
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return {}
    
    def _is_special_function(self, func_name: str) -> bool:
        """Check if a function is special (like __init__, main, etc.)."""
        special_functions = {
            '__init__', '__main__', 'main', 'if __name__ == "__main__"',
            'setup', 'teardown', 'test_', 'run_', 'execute'
        }
        return any(func_name.startswith(special) for special in special_functions)
    
    def _is_special_class(self, class_name: str) -> bool:
        """Check if a class is special (like base classes, etc.)."""
        special_classes = {
            'Base', 'Abstract', 'Interface', 'Protocol', 'Exception', 'Error'
        }
        return any(class_name.startswith(special) for special in special_classes)
    
    def _is_special_method(self, method_name: str) -> bool:
        """Check if a method is special (like __init__, __str__, etc.)."""
        special_methods = {
            '__init__', '__str__', '__repr__', '__len__', '__getitem__',
            '__setitem__', '__delitem__', '__iter__', '__next__', '__enter__',
            '__exit__', '__call__', '__eq__', '__ne__', '__lt__', '__le__',
            '__gt__', '__ge__', '__hash__', '__bool__', '__add__', '__sub__',
            '__mul__', '__div__', '__mod__', '__pow__', '__and__', '__or__',
            '__xor__', '__invert__', '__lshift__', '__rshift__', '__neg__',
            '__pos__', '__abs__', '__round__', '__floor__', '__ceil__',
            '__trunc__', '__index__', '__int__', '__float__', '__complex__',
            '__oct__', '__hex__', '__dir__', '__getattr__', '__setattr__',
            '__delattr__', '__getattribute__', '__new__', '__del__'
        }
        return any(method_name.endswith(special) for special in special_methods)
    
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
    
    def _get_method_line(self, tree: ast.AST, method_name: str) -> int:
        """Get line number of a method definition."""
        class_name, method_name = method_name.split('.', 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == method_name:
                        return child.lineno
        return 0
    
    def analyze_project(self):
        """Analyze the entire project for dead code."""
        print("🔍 Analyzing project for dead code...")
        
        python_files = list(self.project_root.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files")
        
        total_unused_functions = 0
        total_unused_classes = 0
        total_unused_methods = 0
        files_with_dead_code = 0
        
        for file_path in python_files:
            if "test" in str(file_path) or "__pycache__" in str(file_path):
                continue
                
            results = self.analyze_file_for_dead_code(file_path)
            
            if results:
                unused_funcs = results.get("unused_functions", [])
                unused_classes = results.get("unused_classes", [])
                unused_methods = results.get("unused_methods", [])
                
                if unused_funcs or unused_classes or unused_methods:
                    files_with_dead_code += 1
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
                    
                    if unused_methods:
                        print(f"  🔴 Unused methods: {len(unused_methods)}")
                        for method in unused_methods[:3]:  # Show first 3
                            print(f"    - {method['name']} (line {method['line']})")
                        if len(unused_methods) > 3:
                            print(f"    ... and {len(unused_methods) - 3} more")
                
                total_unused_functions += len(unused_funcs)
                total_unused_classes += len(unused_classes)
                total_unused_methods += len(unused_methods)
        
        print(f"\n📊 PROPER DEAD CODE ANALYSIS SUMMARY:")
        print(f"  🔴 Total unused functions: {total_unused_functions}")
        print(f"  🔴 Total unused classes: {total_unused_classes}")
        print(f"  🔴 Total unused methods: {total_unused_methods}")
        print(f"  📁 Files with dead code: {files_with_dead_code}")
        print(f"  📁 Total files analyzed: {len(python_files)}")
        
        return {
            "total_unused_functions": total_unused_functions,
            "total_unused_classes": total_unused_classes,
            "total_unused_methods": total_unused_methods,
            "files_with_dead_code": files_with_dead_code,
            "files_analyzed": len(python_files)
        }

def main():
    project_root = "/Users/remyroche/Documents/Ares"
    analyzer = ProperDeadCodeAnalyzer(project_root)
    
    print("🚀 Starting PROPER Dead Code Analysis using Interaction Mapping Data")
    print("=" * 70)
    
    # Load interaction data
    analyzer.load_interaction_data()
    
    # Analyze project
    results = analyzer.analyze_project()
    
    print("\n✅ Proper dead code analysis complete!")
    if results['files_analyzed'] > 0:
        dead_code_percentage = (results['total_unused_functions'] + results['total_unused_classes']) / results['files_analyzed'] * 100
        print(f"📈 Dead code percentage: {dead_code_percentage:.1f}%")

if __name__ == "__main__":
    main()
