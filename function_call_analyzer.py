#!/usr/bin/env python3
"""
Function Call Analyzer for Python repositories.
Maps function-to-function call relationships and generates call graphs.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import json
import re
from typing import Dict, Set, List, Tuple

class FunctionCallAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.function_calls = defaultdict(set)  # function -> called_functions
        self.function_definitions = defaultdict(str)  # function_name -> file_path
        self.call_graph = defaultdict(set)  # caller -> callees
        self.reverse_call_graph = defaultdict(set)  # callee -> callers
        self.class_methods = defaultdict(set)  # class_name -> methods
        self.syntax_errors = defaultdict(list)
        
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def get_full_function_name(self, node, context=""):
        """Get the full name of a function including class context."""
        if hasattr(node, 'id'):
            if context:
                return f"{context}.{node.id}"
            return node.id
        elif hasattr(node, 'attr'):
            if context:
                return f"{context}.{node.attr}"
            return node.attr
        return "unknown"
    
    def extract_function_calls(self, tree, file_path, context=""):
        """Extract function calls from AST."""
        calls = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Handle different types of function calls
                if isinstance(node.func, ast.Name):
                    # Direct function call: function_name()
                    func_name = node.func.id
                    calls.add(func_name)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: object.method()
                    if isinstance(node.func.value, ast.Name):
                        # obj.method()
                        obj_name = node.func.value.id
                        method_name = node.func.attr
                        calls.add(f"{obj_name}.{method_name}")
                    elif isinstance(node.func.value, ast.Attribute):
                        # obj.attr.method()
                        if isinstance(node.func.value.value, ast.Name):
                            obj_name = node.func.value.value.id
                            attr_name = node.func.value.attr
                            method_name = node.func.attr
                            calls.add(f"{obj_name}.{attr_name}.{method_name}")
                elif isinstance(node.func, ast.Call):
                    # Chained call: function()()
                    if isinstance(node.func.func, ast.Name):
                        calls.add(node.func.func.id)
        
        return calls
    
    def extract_function_definitions(self, tree, file_path):
        """Extract function and method definitions from AST."""
        functions = {}
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Regular function
                func_name = node.name
                functions[func_name] = str(file_path)
                
                # Extract calls within this function
                calls = self.extract_function_calls(node, file_path, func_name)
                if calls:
                    self.function_calls[func_name] = calls
                    
            elif isinstance(node, ast.ClassDef):
                # Class definition
                class_name = node.name
                classes[class_name] = str(file_path)
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Method within class
                        method_name = f"{class_name}.{item.name}"
                        functions[method_name] = str(file_path)
                        
                        # Extract calls within this method
                        calls = self.extract_function_calls(item, file_path, method_name)
                        if calls:
                            self.function_calls[method_name] = calls
        
        return functions, classes
    
    def parse_file(self, file_path):
        """Parse a Python file and extract function information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                functions, classes = self.extract_function_definitions(tree, file_path)
                
                # Store function definitions
                for func_name, file_path_str in functions.items():
                    self.function_definitions[func_name] = file_path_str
                
                # Store class methods
                for class_name, file_path_str in classes.items():
                    self.class_methods[class_name] = file_path_str
                
                return True
                
            except SyntaxError as e:
                self.syntax_errors[file_path].append(f"Syntax error: {e}")
                return False
                
        except Exception as e:
            self.syntax_errors[file_path].append(f"File error: {e}")
            return False
    
    def build_call_graph(self):
        """Build the complete call graph."""
        for caller, calls in self.function_calls.items():
            for callee in calls:
                # Add to call graph
                self.call_graph[caller].add(callee)
                # Add to reverse call graph
                self.reverse_call_graph[callee].add(caller)
    
    def find_call_chains(self, max_depth=3):
        """Find call chains up to a certain depth."""
        chains = []
        
        def explore_chain(caller, depth, current_chain):
            if depth >= max_depth:
                return
            
            if caller in self.call_graph:
                for callee in self.call_graph[caller]:
                    new_chain = current_chain + [callee]
                    chains.append(new_chain)
                    explore_chain(callee, depth + 1, new_chain)
        
        for caller in self.call_graph:
            explore_chain(caller, 0, [caller])
        
        return chains
    
    def find_most_called_functions(self, top_n=10):
        """Find the most frequently called functions."""
        call_counts = defaultdict(int)
        
        for calls in self.function_calls.values():
            for callee in calls:
                call_counts[callee] += 1
        
        return sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def find_most_calling_functions(self, top_n=10):
        """Find functions that call the most other functions."""
        call_counts = {func: len(calls) for func, calls in self.function_calls.items()}
        return sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def generate_dot_call_graph(self, output_path="function_calls.dot"):
        """Generate DOT format call graph."""
        dot_content = ["digraph FunctionCalls {"]
        dot_content.append("  rankdir=LR;")
        dot_content.append("  node [shape=box];")
        
        # Add nodes for all functions
        all_functions = set(self.call_graph.keys()) | set(self.reverse_call_graph.keys())
        for func in all_functions:
            if func in self.function_definitions:
                dot_content.append(f'  "{func}" [label="{func}\\n{self.function_definitions[func]}"];')
            else:
                dot_content.append(f'  "{func}" [label="{func}\\n(undefined)"];')
        
        # Add edges for function calls
        for caller, callees in self.call_graph.items():
            for callee in callees:
                dot_content.append(f'  "{caller}" -> "{callee}";')
        
        dot_content.append("}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(dot_content))
        
        print(f"Function call graph saved to {output_path}")
    
    def generate_json_report(self, output_path="function_calls_report.json"):
        """Generate detailed JSON report."""
        report = {
            "summary": {
                "total_functions": len(self.function_definitions),
                "total_function_calls": sum(len(calls) for calls in self.function_calls.values()),
                "total_classes": len(self.class_methods),
                "files_with_syntax_errors": len(self.syntax_errors)
            },
            "function_definitions": dict(self.function_definitions),
            "function_calls": {k: list(v) for k, v in self.function_calls.items()},
            "call_graph": {k: list(v) for k, v in self.call_graph.items()},
            "reverse_call_graph": {k: list(v) for k, v in self.reverse_call_graph.items()},
            "class_methods": dict(self.class_methods),
            "most_called_functions": self.find_most_called_functions(20),
            "most_calling_functions": self.find_most_calling_functions(20),
            "call_chains": self.find_call_chains(max_depth=3),
            "syntax_errors": {str(k): v for k, v in self.syntax_errors.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Function calls report saved to {output_path}")
    
    def print_summary(self):
        """Print a comprehensive summary."""
        print(f"\n{'='*70}")
        print(f"FUNCTION CALL ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"📁 Total functions defined: {len(self.function_definitions)}")
        print(f"🔗 Total function calls: {sum(len(calls) for calls in self.function_calls.values())}")
        print(f"🏗️  Total classes: {len(self.class_methods)}")
        print(f"⚠️  Files with syntax errors: {len(self.syntax_errors)}")
        
        print(f"\n📊 TOP 10 MOST CALLED FUNCTIONS:")
        most_called = self.find_most_called_functions(10)
        for i, (func, count) in enumerate(most_called, 1):
            print(f"   {i:2d}. {func}: called {count} times")
        
        print(f"\n🔍 TOP 10 MOST CALLING FUNCTIONS:")
        most_calling = self.find_most_calling_functions(10)
        for i, (func, count) in enumerate(most_calling, 1):
            print(f"   {i:2d}. {func}: calls {count} other functions")
        
        print(f"\n🏗️  CLASSES WITH METHODS:")
        for class_name, file_path in list(self.class_methods.items())[:10]:
            methods = [func for func in self.function_definitions.keys() if func.startswith(f"{class_name}.")]
            print(f"   {class_name}: {len(methods)} methods in {file_path}")
        
        if self.syntax_errors:
            print(f"\n⚠️  SYNTAX ERRORS SUMMARY:")
            print(f"   Files with errors: {len(self.syntax_errors)}")
            print(f"   Total errors: {sum(len(errors) for errors in self.syntax_errors.values())}")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Function call analysis for: {root_dir}")
    
    analyzer = FunctionCallAnalyzer(root_dir)
    
    # Parse all files
    python_files = analyzer.find_python_files()
    print(f"Found {len(python_files)} Python files to analyze...")
    
    for file_path in python_files:
        analyzer.parse_file(file_path)
    
    # Build call graph
    analyzer.build_call_graph()
    
    # Generate outputs
    analyzer.generate_dot_call_graph()
    analyzer.generate_json_report()
    
    # Print summary
    analyzer.print_summary()

if __name__ == "__main__":
    main()