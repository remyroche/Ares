#!/usr/bin/env python3
"""
Script to detect and analyze circular imports in the codebase.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path


class ImportAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports = defaultdict(set)  # file -> set of imported modules
        self.module_to_file = {}  # module name -> file path
        self.cycles = []

    def analyze_file(self, file_path: Path) -> set[str]:
        """Extract imports from a Python file."""
        imports = set()

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:  # Absolute imports only
                        imports.add(node.module)
                    elif node.module and node.level > 0:  # Relative imports
                        # Convert relative to absolute
                        module_parts = str(file_path.relative_to(self.project_root)).replace(".py", "").split("/")
                        module_parts = module_parts[:-1]  # Remove file name

                        if node.level == 1:  # from . import
                            base = ".".join(module_parts)
                        else:  # from .. import
                            base = ".".join(module_parts[:-(node.level-1)])

                        if base and node.module:
                            imports.add(f"{base}.{node.module}")
                        elif base:
                            imports.add(base)

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

        return imports

    def build_import_graph(self):
        """Build the import dependency graph."""
        print("Building import graph...")

        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            # Skip __pycache__ and other excluded directories
            if "__pycache__" in str(file_path) or ".venv" in str(file_path):
                continue

            # Get module name from file path
            rel_path = file_path.relative_to(self.project_root)
            module_name = str(rel_path).replace("/", ".").replace(".py", "")

            # Handle __init__.py files
            module_name = module_name.removesuffix(".__init__")  # Remove .__init__

            self.module_to_file[module_name] = str(file_path)

            # Extract imports
            imports = self.analyze_file(file_path)

            # Filter imports to only include project modules
            project_imports = set()
            for imp in imports:
                # Check if it's a project module
                if imp.startswith("src.") or imp in self.module_to_file:
                    project_imports.add(imp)
                elif "." not in imp:
                    # Could be a local module
                    for module in self.module_to_file:
                        if module.endswith(f".{imp}") or module == imp:
                            project_imports.add(module)
                            break

            self.imports[module_name] = project_imports

    def find_cycles_dfs(self):
        """Find circular imports using DFS."""
        visited = set()
        rec_stack = set()
        path = []

        def dfs(module):
            visited.add(module)
            rec_stack.add(module)
            path.append(module)

            for imported in self.imports.get(module, []):
                if imported not in visited:
                    if dfs(imported):
                        return True
                elif imported in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(imported)
                    cycle = path[cycle_start:] + [imported]
                    self.cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(module)
            return False

        # Check all modules
        for module in list(self.imports.keys()):
            if module not in visited:
                dfs(module)

    def find_all_cycles(self):
        """Find all circular imports using a modified DFS approach."""
        all_cycles = []

        def find_cycles_from_node(start_node):
            visited = {start_node}
            stack = [(start_node, [start_node])]

            while stack:
                node, path = stack.pop()

                for neighbor in self.imports.get(node, []):
                    if neighbor == start_node and len(path) > 1:
                        # Found a cycle back to start
                        all_cycles.append(path + [neighbor])
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))

        # Find cycles starting from each node
        for module in self.imports:
            find_cycles_from_node(module)

        # Remove duplicate cycles
        unique_cycles = []
        seen = set()

        for cycle in all_cycles:
            # Normalize cycle (start from smallest element)
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])

            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(list(normalized)[:-1])  # Remove duplicate last element

        self.cycles = unique_cycles

    def analyze_import_depth(self) -> dict[str, int]:
        """Calculate import depth for each module."""
        depths = {}

        def get_depth(module, visited=None):
            if visited is None:
                visited = set()

            if module in depths:
                return depths[module]

            if module in visited:
                return 0  # Circular dependency

            visited.add(module)

            if module not in self.imports or not self.imports[module]:
                depth = 0
            else:
                depth = 1 + max(get_depth(imp, visited.copy())
                              for imp in self.imports[module])

            depths[module] = depth
            return depth

        for module in self.imports:
            get_depth(module)

        return depths

    def generate_report(self) -> dict:
        """Generate a comprehensive circular import report."""
        # Build the import graph
        self.build_import_graph()

        # Find circular imports
        self.find_all_cycles()

        # Analyze import depths
        depths = self.analyze_import_depth()

        # Prepare report
        report = {
            "total_modules": len(self.imports),
            "total_imports": sum(len(imps) for imps in self.imports.values()),
            "circular_imports": {
                "count": len(self.cycles),
                "cycles": [],
            },
            "import_depths": {
                "max_depth": max(depths.values()) if depths else 0,
                "average_depth": sum(depths.values()) / len(depths) if depths else 0,
                "deep_modules": [],
            },
            "highly_imported": [],
            "highly_importing": [],
        }

        # Add cycle details
        for cycle in self.cycles:
            cycle_info = {
                "modules": cycle,
                "length": len(cycle),
                "files": [self.module_to_file.get(m, "Unknown") for m in cycle],
            }
            report["circular_imports"]["cycles"].append(cycle_info)

        # Find modules with deep import chains
        for module, depth in sorted(depths.items(), key=lambda x: x[1], reverse=True)[:10]:
            if depth > 5:
                report["import_depths"]["deep_modules"].append({
                    "module": module,
                    "depth": depth,
                    "file": self.module_to_file.get(module, "Unknown"),
                })

        # Find highly imported modules (many modules import them)
        imported_by = defaultdict(set)
        for module, imports in self.imports.items():
            for imp in imports:
                imported_by[imp].add(module)

        for module, importers in sorted(imported_by.items(),
                                      key=lambda x: len(x[1]), reverse=True)[:10]:
            report["highly_imported"].append({
                "module": module,
                "imported_by_count": len(importers),
                "imported_by": list(importers)[:5],  # Show first 5
            })

        # Find modules that import many others
        for module, imports in sorted(self.imports.items(),
                                    key=lambda x: len(x[1]), reverse=True)[:10]:
            report["highly_importing"].append({
                "module": module,
                "imports_count": len(imports),
                "imports": list(imports)[:5],  # Show first 5
            })

        return report

    def suggest_fixes(self) -> list[dict]:
        """Suggest fixes for circular imports."""
        suggestions = []

        for cycle in self.cycles:
            suggestion = {
                "cycle": cycle,
                "suggestions": [],
            }

            # Analyze the cycle
            if len(cycle) == 2:
                # Direct circular import
                suggestion["suggestions"].append(
                    f"Consider moving shared code from {cycle[0]} and {cycle[1]} "
                    f"to a common module",
                )
            else:
                # Longer cycle
                suggestion["suggestions"].append(
                    "Break the cycle by introducing an interface or abstract base class",
                )
                suggestion["suggestions"].append(
                    "Consider lazy imports using import inside functions",
                )

            # Check for specific patterns
            if any("utils" in m for m in cycle):
                suggestion["suggestions"].append(
                    "Utils modules should not import from feature modules",
                )

            if any("config" in m for m in cycle):
                suggestion["suggestions"].append(
                    "Configuration modules should be leaf nodes with no imports",
                )

            suggestions.append(suggestion)

        return suggestions


def main():
    import argparse
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"/workspace/code_quality/reports/circular_imports_report_{timestamp}.json"

    parser = argparse.ArgumentParser(description="Detect circular imports")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory to analyze")
    parser.add_argument("--output", default=default_output,
                       help="Output report file")

    args = parser.parse_args()

    print("Analyzing circular imports...")
    print("=" * 60)

    analyzer = ImportAnalyzer(args.project_root)
    report = analyzer.generate_report()

    # Print summary
    print(f"\nTotal modules analyzed: {report['total_modules']}")
    print(f"Total import relationships: {report['total_imports']}")
    print(f"Circular imports found: {report['circular_imports']['count']}")

    if report["circular_imports"]["cycles"]:
        print("\nCircular import cycles:")
        for i, cycle in enumerate(report["circular_imports"]["cycles"][:5], 1):
            print(f"\n{i}. Cycle of length {cycle['length']}:")
            for j, module in enumerate(cycle["modules"]):
                if j < len(cycle["modules"]) - 1:
                    print(f"   {module} → {cycle['modules'][j+1]}")
                else:
                    print(f"   {module} → {cycle['modules'][0]}")

    # Get suggestions
    suggestions = analyzer.suggest_fixes()
    if suggestions:
        print("\nSuggested fixes:")
        for i, sugg in enumerate(suggestions[:3], 1):
            print(f"\n{i}. For cycle: {' → '.join(sugg['cycle'][:3])}...")
            for fix in sugg["suggestions"]:
                print(f"   - {fix}")

    # Save full report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
