#!/usr/bin/env python3
"""
Enhanced dependency analyzer for Python repositories.
Handles syntax errors gracefully and provides detailed analysis.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import json
import re

class EnhancedDependencyAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        self.file_imports = defaultdict(set)
        self.syntax_errors = defaultdict(list)
        self.external_imports = defaultdict(set)
        self.standard_lib_imports = defaultdict(set)
        
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip common directories that shouldn't be analyzed
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def get_module_name(self, file_path):
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.root_dir)
        return str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
    
    def is_standard_lib(self, module_name):
        """Check if a module is part of Python standard library."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def parse_imports_with_regex(self, content):
        """Parse imports using regex as fallback when AST fails."""
        imports = set()
        
        # Match import statements
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                match = re.match(import_pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    imports.add(module)
            elif line.startswith('from '):
                match = re.match(from_pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    imports.add(module)
        
        return imports
    
    def parse_imports(self, file_path):
        """Parse imports from a Python file using AST with regex fallback."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try AST parsing first
            try:
                tree = ast.parse(content)
                imports = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                return imports
                
            except SyntaxError as e:
                # Fallback to regex parsing
                self.syntax_errors[file_path].append(f"Syntax error: {e}")
                return self.parse_imports_with_regex(content)
                
        except Exception as e:
            self.syntax_errors[file_path].append(f"File error: {e}")
            return set()
    
    def analyze_dependencies(self):
        """Analyze all dependencies in the repository."""
        python_files = self.find_python_files()
        
        for file_path in python_files:
            module_name = self.get_module_name(file_path)
            imports = self.parse_imports(file_path)
            
            self.file_imports[module_name] = imports
            
            for imported_module in imports:
                self.dependencies[module_name].add(imported_module)
                self.reverse_dependencies[imported_module].add(module_name)
                
                # Categorize imports
                if self.is_standard_lib(imported_module):
                    self.standard_lib_imports[module_name].add(imported_module)
                else:
                    self.external_imports[module_name].add(imported_module)
    
    def generate_dependency_graph(self):
        """Generate a dependency graph in DOT format."""
        dot_content = ["digraph Dependencies {"]
        dot_content.append("  rankdir=LR;")
        dot_content.append("  node [shape=box];")
        
        # Color code nodes based on import types
        for module, deps in self.dependencies.items():
            for dep in deps:
                if dep in self.dependencies:  # Internal dependency
                    dot_content.append(f'  "{module}" -> "{dep}" [color=blue];')
                else:  # External dependency
                    dot_content.append(f'  "{module}" -> "{dep}" [color=red, style=dashed];')
        
        dot_content.append("}")
        return "\n".join(dot_content)
    
    def find_circular_dependencies(self):
        """Find circular dependencies using topological sort."""
        in_degree = defaultdict(int)
        graph = defaultdict(set)
        
        # Build graph and calculate in-degrees
        for module, deps in self.dependencies.items():
            for dep in deps:
                if dep in self.dependencies:  # Only internal dependencies
                    graph[dep].add(module)
                    in_degree[module] += 1
        
        # Topological sort
        queue = deque([module for module in self.dependencies if in_degree[module] == 0])
        topo_order = []
        
        while queue:
            module = queue.popleft()
            topo_order.append(module)
            
            for neighbor in graph[module]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(topo_order) != len(self.dependencies):
            remaining = set(self.dependencies.keys()) - set(topo_order)
            return list(remaining)
        
        return []
    
    def generate_report(self):
        """Generate a comprehensive dependency report."""
        report = {
            "summary": {
                "total_modules": len(self.dependencies),
                "total_imports": sum(len(deps) for deps in self.dependencies.values()),
                "internal_imports": sum(len([d for d in deps if d in self.dependencies]) 
                                      for deps in self.dependencies.values()),
                "external_imports": sum(len([d for d in deps if d not in self.dependencies]) 
                                      for deps in self.dependencies.values()),
                "circular_dependencies": self.find_circular_dependencies(),
                "files_with_syntax_errors": len(self.syntax_errors)
            },
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "reverse_dependencies": {k: list(v) for k, v in self.reverse_dependencies.items()},
            "file_imports": {k: list(v) for k, v in self.file_imports.items()},
            "external_imports": {k: list(v) for k, v in self.external_imports.items()},
            "standard_lib_imports": {k: list(v) for k, v in self.standard_lib_imports.items()},
            "syntax_errors": {str(k): v for k, v in self.syntax_errors.items()}
        }
        return report
    
    def save_dot_file(self, output_path="enhanced_dependencies.dot"):
        """Save dependency graph as DOT file."""
        dot_content = self.generate_dependency_graph()
        with open(output_path, 'w') as f:
            f.write(dot_content)
        print(f"Dependency graph saved to {output_path}")
    
    def save_report(self, output_path="enhanced_dependency_report.json"):
        """Save dependency report as JSON."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Dependency report saved to {output_path}")
    
    def print_summary(self):
        """Print a detailed summary of the analysis."""
        report = self.generate_report()
        summary = report["summary"]
        
        print(f"\n{'='*60}")
        print(f"ENHANCED DEPENDENCY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📁 Total modules analyzed: {summary['total_modules']}")
        print(f"🔗 Total imports found: {summary['total_imports']}")
        print(f"🏠 Internal imports: {summary['internal_imports']}")
        print(f"🌐 External imports: {summary['external_imports']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        
        if summary['circular_dependencies']:
            print(f"🔄 Circular dependencies found: {len(summary['circular_dependencies'])}")
            for dep in summary['circular_dependencies'][:5]:  # Show first 5
                print(f"   - {dep}")
        else:
            print("✅ No circular dependencies detected")
        
        print(f"\n📊 TOP 10 MOST IMPORTED MODULES:")
        sorted_reverse = sorted(self.reverse_dependencies.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        for i, (module, importers) in enumerate(sorted_reverse[:10], 1):
            print(f"   {i:2d}. {module}: imported by {len(importers)} modules")
        
        print(f"\n🔍 TOP 10 MOST DEPENDENT MODULES:")
        sorted_deps = sorted(self.dependencies.items(), 
                            key=lambda x: len(x[1]), reverse=True)
        for i, (module, deps) in enumerate(sorted_deps[:10], 1):
            print(f"   {i:2d}. {module}: imports {len(deps)} modules")
        
        if self.syntax_errors:
            print(f"\n⚠️  SYNTAX ERRORS SUMMARY:")
            print(f"   Files with errors: {len(self.syntax_errors)}")
            print(f"   Total errors: {sum(len(errors) for errors in self.syntax_errors.values())}")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Enhanced dependency analysis for: {root_dir}")
    
    analyzer = EnhancedDependencyAnalyzer(root_dir)
    analyzer.analyze_dependencies()
    
    # Generate outputs
    analyzer.save_dot_file()
    analyzer.save_report()
    
    # Print detailed summary
    analyzer.print_summary()

if __name__ == "__main__":
    main()