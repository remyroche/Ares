"""
Import Analysis - Detects import conflicts, duplicates, and circular dependencies.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, deque
import networkx as nx

from ..core.config import CodeQualityConfig


class ImportIssue:
    """Represents an import-related issue."""
    
    def __init__(self, file_path: str, line_number: int, issue_type: str, 
                 message: str, severity: str = "warning", details: Optional[Dict] = None):
        self.file_path = file_path
        self.line_number = line_number
        self.issue_type = issue_type
        self.message = message
        self.severity = severity
        self.details = details or {}


class ImportAnalyzer:
    """Analyzes Python imports for conflicts, duplicates, and circular dependencies."""
    
    def __init__(self, config: CodeQualityConfig):
        self.config = config
        self.import_graph = nx.DiGraph()
        self.imports_by_file = defaultdict(list)
        self.duplicate_imports = []
        self.circular_dependencies = []
        self.unused_imports = []
        self.conflicting_imports = []
        
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze imports in all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.analysis.exclude_patterns]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return self.analyze_files(python_files)
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze imports in specific Python files."""
        print(f"Analyzing imports in {len(file_paths)} files...")
        
        # First pass: collect all imports and build import graph
        for file_path in file_paths:
            try:
                self._analyze_file_imports(file_path)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        # Second pass: detect issues
        self._detect_duplicate_imports()
        self._detect_circular_dependencies()
        self._detect_unused_imports()
        self._detect_conflicting_imports()
        
        return self._generate_report()
    
    def _analyze_file_imports(self, file_path: str) -> None:
        """Analyze imports in a single file and add to import graph."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Collect imports from this file
            file_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.name
                        as_name = alias.asname or import_name
                        
                        file_imports.append({
                            'type': 'import',
                            'module': import_name,
                            'as_name': as_name,
                            'line': node.lineno
                        })
                        
                        # Add to import graph
                        self.import_graph.add_edge(file_path, import_name)
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        import_name = f"{module}.{alias.name}" if module else alias.name
                        as_name = alias.asname or alias.name
                        
                        file_imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'full_name': import_name,
                            'as_name': as_name,
                            'line': node.lineno
                        })
                        
                        # Add to import graph
                        if module:
                            self.import_graph.add_edge(file_path, module)
            
            self.imports_by_file[file_path] = file_imports
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    def _detect_duplicate_imports(self) -> None:
        """Detect duplicate imports within files and across files."""
        # Check for duplicates within each file
        for file_path, imports in self.imports_by_file.items():
            seen_imports = set()
            
            for imp in imports:
                if imp['type'] == 'import':
                    key = (imp['module'], imp['as_name'])
                else:  # from_import
                    key = (imp['module'], imp['name'], imp['as_name'])
                
                if key in seen_imports:
                    self.duplicate_imports.append(ImportIssue(
                        file_path=file_path,
                        line_number=imp['line'],
                        issue_type='duplicate_import',
                        message=f"Duplicate import: {imp.get('full_name', imp.get('module'))}",
                        severity='warning',
                        details={'import_info': imp}
                    ))
                else:
                    seen_imports.add(key)
        
        # Check for duplicate imports across files (same module imported differently)
        module_imports = defaultdict(list)
        for file_path, imports in self.imports_by_file.items():
            for imp in imports:
                if imp['type'] == 'import':
                    module_imports[imp['module']].append((file_path, imp))
                else:  # from_import
                    module_imports[imp['full_name']].append((file_path, imp))
        
        for module, import_list in module_imports.items():
            if len(import_list) > 1:
                # Check if they're imported consistently
                first_import = import_list[0][1]
                for file_path, imp in import_list[1:]:
                    if self._imports_are_inconsistent(first_import, imp):
                        self.duplicate_imports.append(ImportIssue(
                            file_path=file_path,
                            line_number=imp['line'],
                            issue_type='inconsistent_import',
                            message=f"Inconsistent import of {module} across files",
                            severity='warning',
                            details={
                                'module': module,
                                'other_imports': [f"{f}:{i['line']}" for f, i in import_list]
                            }
                        ))
    
    def _imports_are_inconsistent(self, imp1: Dict, imp2: Dict) -> bool:
        """Check if two imports of the same module are inconsistent."""
        if imp1['type'] != imp2['type']:
            return True
        
        if imp1['type'] == 'import':
            return imp1['as_name'] != imp2['as_name']
        else:  # from_import
            return (imp1['module'] != imp2['module'] or 
                   imp1['name'] != imp2['name'] or 
                   imp1['as_name'] != imp2['as_name'])
    
    def _detect_circular_dependencies(self) -> None:
        """Detect circular dependencies in the import graph."""
        try:
            cycles = list(nx.simple_cycles(self.import_graph))
            
            for cycle in cycles:
                if len(cycle) > 1:  # Ignore self-imports
                    self.circular_dependencies.append({
                        'cycle': cycle,
                        'files': [node for node in cycle if os.path.exists(node)],
                        'modules': [node for node in cycle if not os.path.exists(node)]
                    })
                    
                    # Create import issues for each file in the cycle
                    for file_path in cycle:
                        if os.path.exists(file_path):
                            self.circular_dependencies.append(ImportIssue(
                                file_path=file_path,
                                line_number=0,  # We don't have specific line info for cycles
                                issue_type='circular_dependency',
                                message=f"Part of circular dependency: {' -> '.join(cycle)}",
                                severity='error',
                                details={'cycle': cycle}
                            ))
        except Exception as e:
            print(f"Error detecting circular dependencies: {e}")
    
    def _detect_unused_imports(self) -> None:
        """Detect potentially unused imports (basic heuristic)."""
        # This is a simplified check - for more accurate results, you'd need to
        # analyze actual usage patterns in the code
        
        for file_path, imports in self.imports_by_file.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for imp in imports:
                    if imp['type'] == 'import':
                        # Check if the imported name is used
                        imported_name = imp['as_name']
                        if imported_name not in content.replace(imp['module'], ''):
                            # This is a very basic check - could have false positives
                            pass
                    else:  # from_import
                        # Check if the imported name is used
                        imported_name = imp['as_name']
                        if imported_name not in content:
                            # This is a very basic check - could have false positives
                            pass
                            
            except Exception as e:
                print(f"Error checking unused imports in {file_path}: {e}")
    
    def _detect_conflicting_imports(self) -> None:
        """Detect conflicting imports (e.g., same name imported from different modules)."""
        name_sources = defaultdict(list)
        
        for file_path, imports in self.imports_by_file.items():
            for imp in imports:
                if imp['type'] == 'import':
                    name_sources[imp['as_name']].append({
                        'file': file_path,
                        'module': imp['module'],
                        'line': imp['line']
                    })
                else:  # from_import
                    name_sources[imp['as_name']].append({
                        'file': file_path,
                        'module': imp['module'],
                        'name': imp['name'],
                        'line': imp['line']
                    })
        
        # Check for conflicts
        for name, sources in name_sources.items():
            if len(sources) > 1:
                # Check if the same name comes from different modules
                modules = set()
                for source in sources:
                    if source['module']:
                        modules.add(source['module'])
                
                if len(modules) > 1:
                    for source in sources:
                        self.conflicting_imports.append(ImportIssue(
                            file_path=source['file'],
                            line_number=source['line'],
                            issue_type='conflicting_import',
                            message=f"Name '{name}' conflicts with imports from other modules: {', '.join(modules)}",
                            severity='warning',
                            details={
                                'name': name,
                                'conflicting_modules': list(modules),
                                'all_sources': sources
                            }
                        ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive import analysis report."""
        total_issues = (len(self.duplicate_imports) + 
                       len(self.circular_dependencies) + 
                       len(self.conflicting_imports))
        
        return {
            "summary": {
                "total_files_analyzed": len(self.imports_by_file),
                "total_imports": sum(len(imports) for imports in self.imports_by_file.values()),
                "total_issues": total_issues,
                "duplicate_imports": len(self.duplicate_imports),
                "circular_dependencies": len(self.circular_dependencies),
                "conflicting_imports": len(self.conflicting_imports)
            },
            "issues": {
                "duplicate_imports": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details
                    }
                    for issue in self.duplicate_imports
                ],
                "circular_dependencies": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details
                    }
                    for issue in self.circular_dependencies if hasattr(issue, 'file_path')
                ],
                "conflicting_imports": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "details": issue.details
                    }
                    for issue in self.conflicting_imports
                ]
            },
            "import_graph": {
                "nodes": list(self.import_graph.nodes()),
                "edges": list(self.import_graph.edges()),
                "has_cycles": len(list(nx.simple_cycles(self.import_graph))) > 0
            },
            "files": {
                file_path: {
                    "total_imports": len(imports),
                    "import_types": {
                        'import': len([i for i in imports if i['type'] == 'import']),
                        'from_import': len([i for i in imports if i['type'] == 'from_import'])
                    }
                }
                for file_path, imports in self.imports_by_file.items()
            }
        }
    
    def get_import_graph(self) -> nx.DiGraph:
        """Get the import dependency graph."""
        return self.import_graph
    
    def visualize_import_graph(self, output_path: str = None) -> None:
        """Visualize the import dependency graph."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.import_graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.import_graph, pos, node_color='lightblue', 
                                 node_size=1000)
            
            # Draw edges
            nx.draw_networkx_edges(self.import_graph, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(self.import_graph, pos, font_size=8)
            
            plt.title("Import Dependency Graph")
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Import graph saved to: {output_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for graph visualization")
        except Exception as e:
            print(f"Error visualizing import graph: {e}")


def analyze_imports(directory_path: str, config: CodeQualityConfig) -> Dict[str, Any]:
    """Convenience function to analyze imports in a directory."""
    analyzer = ImportAnalyzer(config)
    return analyzer.analyze_directory(directory_path)