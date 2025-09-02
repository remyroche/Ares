#!/usr/bin/env python3
"""
Comprehensive Unused Code Analyzer for Python repositories.
Identifies unused modules, functions, classes, and imports.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import json
import re
from typing import Dict, Set, List, Tuple, Any

class UnusedCodeAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.all_functions = defaultdict(str)  # function_name -> file_path
        self.all_classes = defaultdict(str)   # class_name -> file_path
        self.all_modules = set()              # all module names
        self.imported_modules = set()         # modules that are imported
        self.imported_functions = set()       # functions that are imported
        self.imported_classes = set()         # classes that are imported
        self.function_calls = defaultdict(set)  # function -> called_functions
        self.class_instantiations = defaultdict(set)  # class -> instantiating_files
        self.syntax_errors = defaultdict(list)
        self.unused_functions = set()
        self.unused_classes = set()
        self.unused_modules = set()
        self.unused_imports = defaultdict(set)
        
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def get_module_name(self, file_path):
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.root_dir)
        return str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
    
    def extract_imports(self, tree, file_path):
        """Extract all imports from AST."""
        imports = set()
        from_imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    imports.add(module_name)
                    if alias.asname:
                        imports.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    imports.add(module_name)
                    for alias in node.names:
                        if alias.asname:
                            imports.add(alias.asname)
                        else:
                            from_imports.add(alias.name)
        
        return imports, from_imports
    
    def extract_function_definitions(self, tree, file_path):
        """Extract function definitions from AST."""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                functions[func_name] = str(file_path)
                
                # Extract calls within this function
                calls = self.extract_function_calls(node)
                if calls:
                    self.function_calls[func_name] = calls
                    
        return functions
    
    def extract_class_definitions(self, tree, file_path):
        """Extract class definitions from AST."""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                classes[class_name] = str(file_path)
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = f"{class_name}.{item.name}"
                        classes[method_name] = str(file_path)
                        
                        # Extract calls within this method
                        calls = self.extract_function_calls(item)
                        if calls:
                            self.function_calls[method_name] = calls
                            
        return classes
    
    def extract_function_calls(self, node):
        """Extract function calls from a node."""
        calls = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        calls.add(f"{child.func.value.id}.{child.func.attr}")
        
        return calls
    
    def extract_class_usage(self, tree, file_path):
        """Extract class instantiations and usage."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Direct class instantiation: ClassName()
                    class_name = node.func.id
                    if class_name in self.all_classes:
                        self.class_instantiations[class_name].add(str(file_path))
                elif isinstance(node.func, ast.Attribute):
                    # Method call or attribute access
                    if isinstance(node.func.value, ast.Name):
                        obj_name = node.func.value.id
                        attr_name = node.func.attr
                        # Check if this is a class method call
                        if f"{obj_name}.{attr_name}" in self.all_classes:
                            self.class_instantiations[f"{obj_name}.{attr_name}"].add(str(file_path))
    
    def parse_file(self, file_path):
        """Parse a Python file and extract all information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                
                # Extract imports
                imports, from_imports = self.extract_imports(tree, file_path)
                self.imported_modules.update(imports)
                self.imported_functions.update(from_imports)
                
                # Extract function definitions
                functions = self.extract_function_definitions(tree, file_path)
                for func_name, file_path_str in functions.items():
                    self.all_functions[func_name] = file_path_str
                
                # Extract class definitions
                classes = self.extract_class_definitions(tree, file_path)
                for class_name, file_path_str in classes.items():
                    self.all_classes[class_name] = file_path_str
                
                # Extract class usage
                self.extract_class_usage(tree, file_path)
                
                return True
                
            except SyntaxError as e:
                self.syntax_errors[file_path].append(f"Syntax error: {e}")
                return False
                
        except Exception as e:
            self.syntax_errors[file_path].append(f"File error: {e}")
            return False
    
    def analyze_usage(self):
        """Analyze what code is actually used."""
        # Find unused functions
        used_functions = set()
        for calls in self.function_calls.values():
            used_functions.update(calls)
        
        # Functions that are imported but not defined locally
        imported_but_not_local = self.imported_functions - set(self.all_functions.keys())
        used_functions.update(imported_but_not_local)
        
        # Functions that are never called
        self.unused_functions = set(self.all_functions.keys()) - used_functions
        
        # Find unused classes
        used_classes = set()
        for instantiations in self.class_instantiations.values():
            used_classes.update(instantiations)
        
        # Classes that are imported but not defined locally
        imported_but_not_local_classes = set()  # Could be enhanced
        used_classes.update(imported_but_not_local_classes)
        
        # Classes that are never instantiated
        self.unused_classes = set(self.all_classes.keys()) - used_classes
        
        # Find unused modules
        defined_modules = {self.get_module_name(Path(path)) for path in self.all_functions.values()}
        self.unused_modules = defined_modules - self.imported_modules
    
    def find_dead_imports(self):
        """Find imports that are never used."""
        for file_path in self.find_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports, from_imports = self.extract_imports(tree, file_path)
                
                # Check if imported items are actually used
                used_items = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        used_items.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        if isinstance(node.attr, str):
                            used_items.add(node.attr)
                
                unused_imports = imports - used_items
                if unused_imports:
                    self.unused_imports[file_path] = unused_imports
                    
            except:
                continue
    
    def generate_report(self):
        """Generate comprehensive unused code report."""
        report = {
            "summary": {
                "total_functions": len(self.all_functions),
                "total_classes": len(self.all_classes),
                "unused_functions": len(self.unused_functions),
                "unused_classes": len(self.unused_classes),
                "unused_modules": len(self.unused_modules),
                "files_with_syntax_errors": len(self.syntax_errors),
                "unused_imports_files": len(self.unused_imports)
            },
            "unused_functions": [
                {
                    "name": func,
                    "file": self.all_functions[func]
                } for func in sorted(self.unused_functions)
            ],
            "unused_classes": [
                {
                    "name": class_name,
                    "file": self.all_classes[class_name]
                } for class_name in sorted(self.unused_classes)
            ],
            "unused_modules": list(sorted(self.unused_modules)),
            "unused_imports": {
                str(k): list(v) for k, v in self.unused_imports.items()
            },
            "syntax_errors": {str(k): v for k, v in self.syntax_errors.items()},
            "function_calls": {k: list(v) for k, v in self.function_calls.items()},
            "class_instantiations": {k: list(v) for k, v in self.class_instantiations.items()}
        }
        return report
    
    def save_report(self, output_path="unused_code_report.json"):
        """Save report as JSON."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Unused code report saved to {output_path}")
    
    def print_summary(self):
        """Print comprehensive summary."""
        report = self.generate_report()
        summary = report["summary"]
        
        print(f"\n{'='*80}")
        print(f"UNUSED CODE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"📁 Total functions analyzed: {summary['total_functions']}")
        print(f"🏗️  Total classes analyzed: {summary['total_classes']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        
        print(f"\n🗑️  UNUSED CODE IDENTIFIED:")
        print(f"   • Unused functions: {summary['unused_functions']}")
        print(f"   • Unused classes: {summary['unused_classes']}")
        print(f"   • Unused modules: {summary['unused_modules']}")
        print(f"   • Files with unused imports: {summary['unused_imports_files']}")
        
        if self.unused_functions:
            print(f"\n🔍 TOP 20 UNUSED FUNCTIONS:")
            for i, func_info in enumerate(report['unused_functions'][:20], 1):
                print(f"   {i:2d}. {func_info['name']} in {func_info['file']}")
        
        if self.unused_classes:
            print(f"\n🏗️  TOP 20 UNUSED CLASSES:")
            for i, class_info in enumerate(report['unused_classes'][:20], 1):
                print(f"   {i:2d}. {class_info['name']} in {class_info['file']}")
        
        if self.unused_modules:
            print(f"\n📦 UNUSED MODULES:")
            for module in sorted(self.unused_modules):
                print(f"   • {module}")
        
        if self.unused_imports:
            print(f"\n📥 FILES WITH UNUSED IMPORTS:")
            for file_path, imports in list(self.unused_imports.items())[:10]:
                print(f"   • {file_path}: {', '.join(imports)}")
        
        # Calculate cleanup potential
        total_code = summary['total_functions'] + summary['total_classes']
        unused_code = summary['unused_functions'] + summary['unused_classes']
        if total_code > 0:
            cleanup_percentage = (unused_code / total_code) * 100
            print(f"\n💡 CLEANUP POTENTIAL:")
            print(f"   • {cleanup_percentage:.1f}% of code could potentially be removed")
            print(f"   • {unused_code} unused functions/classes identified")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Unused code analysis for: {root_dir}")
    
    analyzer = UnusedCodeAnalyzer(root_dir)
    
    # Parse all files
    python_files = analyzer.find_python_files()
    print(f"Found {len(python_files)} Python files to analyze...")
    
    for i, file_path in enumerate(python_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(python_files)}...")
        analyzer.parse_file(file_path)
    
    print("Analyzing code usage...")
    analyzer.analyze_usage()
    
    print("Finding dead imports...")
    analyzer.find_dead_imports()
    
    # Generate outputs
    analyzer.save_report()
    
    # Print summary
    analyzer.print_summary()

if __name__ == "__main__":
    main()