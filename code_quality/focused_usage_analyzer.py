#!/usr/bin/env python3
"""
Focused Usage Analyzer - Identifies what code is actually being used
vs. what's just defined but never called.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import json
import re

class FocusedUsageAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.defined_functions = defaultdict(str)  # function -> file
        self.defined_classes = defaultdict(str)   # class -> file
        self.actually_called_functions = set()    # functions that are actually called
        self.actually_used_classes = set()        # classes that are actually used
        self.import_statements = defaultdict(set) # file -> imports
        self.syntax_errors = defaultdict(list)
        self.entry_points = set()                 # potential entry points
        
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def is_entry_point(self, file_path):
        """Check if a file might be an entry point."""
        filename = file_path.name.lower()
        entry_indicators = [
            'main.py', '__main__.py', 'run.py', 'start.py', 'launch.py',
            'app.py', 'server.py', 'cli.py', 'command.py', 'script.py'
        ]
        return any(indicator in filename for indicator in entry_indicators)
    
    def has_main_block(self, content):
        """Check if file has a main execution block."""
        return 'if __name__ == "__main__"' in content
    
    def extract_imports(self, tree, file_path):
        """Extract imports from AST."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        if alias.asname:
                            imports.add(alias.asname)
                        else:
                            imports.add(alias.name)
        
        return imports
    
    def extract_function_definitions(self, tree, file_path):
        """Extract function definitions from AST."""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                functions[func_name] = str(file_path)
                
                # Check if this function calls other functions
                calls = self.extract_function_calls(node)
                if calls:
                    self.actually_called_functions.update(calls)
                    
        return functions
    
    def extract_class_definitions(self, tree, file_path):
        """Extract class definitions from AST."""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                classes[class_name] = str(file_path)
                
                # Check if this class is used (has methods called)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = f"{class_name}.{item.name}"
                        classes[method_name] = str(file_path)
                        
                        # Check if methods call other functions
                        calls = self.extract_function_calls(item)
                        if calls:
                            self.actually_called_functions.update(calls)
                            
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
                    if class_name in self.defined_classes:
                        self.actually_used_classes.add(class_name)
                elif isinstance(node.func, ast.Attribute):
                    # Method call or attribute access
                    if isinstance(node.func.value, ast.Name):
                        obj_name = node.func.value.id
                        attr_name = node.func.attr
                        # Check if this is a class method call
                        if f"{obj_name}.{attr_name}" in self.defined_classes:
                            self.actually_used_classes.add(f"{obj_name}.{attr_name}")
    
    def parse_file(self, file_path):
        """Parse a Python file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if this might be an entry point
            if self.is_entry_point(file_path) or self.has_main_block(content):
                self.entry_points.add(str(file_path))
            
            try:
                tree = ast.parse(content)
                
                # Extract imports
                imports = self.extract_imports(tree, file_path)
                self.import_statements[file_path] = imports
                
                # Extract function definitions
                functions = self.extract_function_definitions(tree, file_path)
                for func_name, file_path_str in functions.items():
                    self.defined_functions[func_name] = file_path_str
                
                # Extract class definitions
                classes = self.extract_class_definitions(tree, file_path)
                for class_name, file_path_str in classes.items():
                    self.defined_classes[class_name] = file_path_str
                
                # Extract class usage
                self.extract_class_usage(tree, file_path)
                
                return True
                
            except SyntaxError as e:
                self.syntax_errors[file_path].append(f"Syntax error: {e}")
                return False
                
        except Exception as e:
            self.syntax_errors[file_path].append(f"File error: {e}")
            return False
    
    def analyze_usage_patterns(self):
        """Analyze usage patterns and identify truly unused code."""
        # Functions that are defined but never called
        truly_unused_functions = set(self.defined_functions.keys()) - self.actually_called_functions
        
        # Classes that are defined but never used
        truly_unused_classes = set(self.defined_classes.keys()) - self.actually_used_classes
        
        # Functions that might be entry points (main, run, etc.)
        potential_entry_functions = {name for name in self.defined_functions.keys() 
                                   if any(keyword in name.lower() for keyword in 
                                         ['main', 'run', 'start', 'execute', 'launch'])}
        
        # Classes that might be entry points
        potential_entry_classes = {name for name in self.defined_classes.keys() 
                                 if any(keyword in name.lower() for keyword in 
                                       ['main', 'app', 'server', 'controller', 'manager'])}
        
        return {
            'truly_unused_functions': truly_unused_functions,
            'truly_unused_classes': truly_unused_classes,
            'potential_entry_functions': potential_entry_functions,
            'potential_entry_classes': potential_entry_classes,
            'entry_points': self.entry_points
        }
    
    def generate_focused_report(self):
        """Generate a focused report on truly unused code."""
        usage_patterns = self.analyze_usage_patterns()
        
        report = {
            "summary": {
                "total_files_analyzed": len(self.find_python_files()),
                "total_functions_defined": len(self.defined_functions),
                "total_classes_defined": len(self.defined_classes),
                "truly_unused_functions": len(usage_patterns['truly_unused_functions']),
                "truly_unused_classes": len(usage_patterns['truly_unused_classes']),
                "potential_entry_points": len(usage_patterns['entry_points']),
                "files_with_syntax_errors": len(self.syntax_errors)
            },
            "truly_unused_functions": [
                {
                    "name": func,
                    "file": self.defined_functions[func]
                } for func in sorted(usage_patterns['truly_unused_functions'])
            ],
            "truly_unused_classes": [
                {
                    "name": class_name,
                    "file": self.defined_classes[class_name]
                } for class_name in sorted(usage_patterns['truly_unused_classes'])
            ],
            "potential_entry_functions": list(usage_patterns['potential_entry_functions']),
            "potential_entry_classes": list(usage_patterns['potential_entry_classes']),
            "entry_points": list(usage_patterns['entry_points']),
            "syntax_errors": {str(k): v for k, v in self.syntax_errors.items()}
        }
        
        return report
    
    def save_report(self, output_path="focused_usage_report.json"):
        """Save focused report as JSON."""
        report = self.generate_focused_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Focused usage report saved to {output_path}")
    
    def print_focused_summary(self):
        """Print focused summary."""
        report = self.generate_focused_report()
        summary = report["summary"]
        
        print(f"\n{'='*80}")
        print(f"FOCUSED USAGE ANALYSIS - TRULY UNUSED CODE")
        print(f"{'='*80}")
        print(f"📁 Total files analyzed: {summary['total_files_analyzed']}")
        print(f"🔧 Total functions defined: {summary['total_functions_defined']}")
        print(f"🏗️  Total classes defined: {summary['total_classes_defined']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        
        print(f"\n🗑️  TRULY UNUSED CODE:")
        print(f"   • Truly unused functions: {summary['truly_unused_functions']}")
        print(f"   • Truly unused classes: {summary['truly_unused_classes']}")
        print(f"   • Total unused code: {summary['truly_unused_functions'] + summary['truly_unused_classes']}")
        
        print(f"\n🚪 POTENTIAL ENTRY POINTS:")
        print(f"   • Entry point files: {summary['potential_entry_points']}")
        print(f"   • Potential entry functions: {len(report['potential_entry_functions'])}")
        print(f"   • Potential entry classes: {len(report['potential_entry_classes'])}")
        
        if report['truly_unused_functions']:
            print(f"\n🔍 TOP 20 TRULY UNUSED FUNCTIONS:")
            for i, func_info in enumerate(report['truly_unused_functions'][:20], 1):
                print(f"   {i:2d}. {func_info['name']} in {func_info['file']}")
        
        if report['truly_unused_classes']:
            print(f"\n🏗️  TOP 20 TRULY UNUSED CLASSES:")
            for i, class_info in enumerate(report['truly_unused_classes'][:20], 1):
                print(f"   {i:2d}. {class_info['name']} in {class_info['file']}")
        
        if report['entry_points']:
            print(f"\n🚪 ENTRY POINT FILES:")
            for entry_point in sorted(report['entry_points']):
                print(f"   • {entry_point}")
        
        # Calculate cleanup potential
        total_code = summary['total_functions_defined'] + summary['total_classes_defined']
        unused_code = summary['truly_unused_functions'] + summary['truly_unused_classes']
        if total_code > 0:
            cleanup_percentage = (unused_code / total_code) * 100
            print(f"\n💡 CLEANUP POTENTIAL:")
            print(f"   • {cleanup_percentage:.1f}% of code is truly unused")
            print(f"   • {unused_code} functions/classes can be safely removed")
            print(f"   • Focus on files without syntax errors first")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Focused usage analysis for: {root_dir}")
    
    analyzer = FocusedUsageAnalyzer(root_dir)
    
    # Parse all files
    python_files = analyzer.find_python_files()
    print(f"Found {len(python_files)} Python files to analyze...")
    
    for i, file_path in enumerate(python_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(python_files)}...")
        analyzer.parse_file(file_path)
    
    print("Analyzing usage patterns...")
    
    # Generate outputs
    analyzer.save_report()
    
    # Print focused summary
    analyzer.print_focused_summary()

if __name__ == "__main__":
    main()