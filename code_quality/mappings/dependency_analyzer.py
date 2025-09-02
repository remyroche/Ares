#!/usr/bin/env python3
"""
Simple dependency analyzer for Python repositories.
Uses Python's built-in ast module to parse imports without external dependencies.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import json

class DependencyAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        self.file_imports = defaultdict(set)
        
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
    
    def parse_imports(self, file_path):
        """Parse imports from a Python file using ast."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
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
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
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
    
    def generate_dependency_graph(self):
        """Generate a dependency graph in DOT format."""
        dot_content = ["digraph Dependencies {"]
        dot_content.append("  rankdir=LR;")
        dot_content.append("  node [shape=box];")
        
        for module, deps in self.dependencies.items():
            for dep in deps:
                dot_content.append(f'  "{module}" -> "{dep}";')
        
        dot_content.append("}")
        return "\n".join(dot_content)
    
    def find_circular_dependencies(self):
        """Find circular dependencies using topological sort."""
        in_degree = defaultdict(int)
        graph = defaultdict(set)
        
        # Build graph and calculate in-degrees
        for module, deps in self.dependencies.items():
            for dep in deps:
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
                "circular_dependencies": self.find_circular_dependencies()
            },
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "reverse_dependencies": {k: list(v) for k, v in self.reverse_dependencies.items()},
            "file_imports": {k: list(v) for k, v in self.file_imports.items()}
        }
        return report
    
    def save_dot_file(self, output_path="dependencies.dot"):
        """Save dependency graph as DOT file."""
        dot_content = self.generate_dependency_graph()
        with open(output_path, 'w') as f:
            f.write(dot_content)
        print(f"Dependency graph saved to {output_path}")
    
    def save_report(self, output_path="dependency_report.json"):
        """Save dependency report as JSON."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Dependency report saved to {output_path}")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"Analyzing dependencies in: {root_dir}")
    
    analyzer = DependencyAnalyzer(root_dir)
    analyzer.analyze_dependencies()
    
    # Generate outputs
    analyzer.save_dot_file()
    analyzer.save_report()
    
    # Print summary
    report = analyzer.generate_report()
    summary = report["summary"]
    
    print(f"\nDependency Analysis Summary:")
    print(f"Total modules: {summary['total_modules']}")
    print(f"Total imports: {summary['total_imports']}")
    
    if summary['circular_dependencies']:
        print(f"⚠️  Circular dependencies found: {summary['circular_dependencies']}")
    else:
        print("✅ No circular dependencies detected")
    
    print(f"\nTop 10 most imported modules:")
    sorted_reverse = sorted(analyzer.reverse_dependencies.items(), 
                           key=lambda x: len(x[1]), reverse=True)
    for module, importers in sorted_reverse[:10]:
        print(f"  {module}: imported by {len(importers)} modules")

if __name__ == "__main__":
    main()