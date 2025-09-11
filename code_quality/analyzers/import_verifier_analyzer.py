#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Import Verifier Analyzer - Checks if each file is imported by others.

This analyzer examines all Python files in a directory and determines whether
each file is imported by any other files, providing a simple yes/no answer
for each file's import status.
"""

import ast
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from .base_analyzer import BaseAnalyzer


class ImportVerifierAnalyzer(BaseAnalyzer):
    """Analyzer to verify if each file is imported by others."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Analyze directory to determine which files are imported by others.
        Always checks the entire repository scope for imports, not just the target directory.
        
        Returns:
            Dict containing import status for each file with yes/no answers
        """
        # Get all Python files in the target directory (files to analyze)
        target_python_files = self._find_python_files(directory_path)
        
        # Get all Python files in the entire repository (for import checking)
        # We need to find the repository root first
        repo_root = self._find_repository_root(directory_path)
        all_python_files = self._find_python_files(repo_root)
        
        # Build a mapping of module names to file paths for the entire repository
        module_to_file = {}
        file_to_module = {}
        
        # First pass: collect all modules and their file paths from entire repo
        for file_path in all_python_files:
            module_name = self._get_module_name(file_path, repo_root)
            module_to_file[module_name] = file_path
            file_to_module[str(file_path)] = module_name
        
        # Second pass: collect all imports from each file in the entire repository
        file_imports = {}
        for file_path in all_python_files:
            content = self._read_file_safely(file_path)
            if not content:
                continue
            
            tree = self._parse_ast_safely(content, file_path)
            if not tree:
                continue
            
            imports = self._extract_all_imports(tree, file_path, repo_root)
            file_imports[str(file_path)] = imports
            self.stats["files_analyzed"] += 1
            self.stats["total_items"] += len(imports)
        
        # Third pass: determine which target files are imported by others (anywhere in repo)
        import_status = {}
        for file_path in target_python_files:
            file_path_str = str(file_path)
            # Try both absolute and relative paths
            module_name = file_to_module.get(file_path_str)
            if not module_name:
                # Try with absolute path
                abs_path = str(file_path.resolve())
                module_name = file_to_module.get(abs_path)
            
            if not module_name:
                import_status[file_path_str] = {
                    "is_imported": False,
                    "imported_by": [],
                    "reason": "Could not determine module name"
                }
                continue
            
            # Check if this module is imported by any other file in the entire repository
            imported_by = []
            for other_file, other_imports in file_imports.items():
                if other_file != file_path_str:  # Don't count self-imports
                    if self._is_module_imported(module_name, other_imports):
                        imported_by.append(other_file)
            
            is_imported = len(imported_by) > 0
            
            # Check if file is only imported by non-production files
            only_imported_by_non_production = self._is_only_imported_by_non_production(imported_by)
            
            import_status[file_path_str] = {
                "is_imported": is_imported,
                "imported_by": imported_by,
                "import_count": len(imported_by),
                "module_name": module_name,
                "only_imported_by_non_production": only_imported_by_non_production
            }
        
        # Calculate summary statistics
        total_files = len(target_python_files)
        imported_files = sum(1 for status in import_status.values() if status["is_imported"])
        unimported_files = total_files - imported_files
        only_non_production_files = sum(1 for status in import_status.values() if status.get("only_imported_by_non_production", False))
        
        # Advanced analysis
        circular_imports = self._detect_circular_imports(import_status, file_to_module)
        import_depths = self._calculate_import_depths(import_status, file_to_module)
        critical_paths = self._identify_critical_paths(import_status)
        
        # Add import depths to import status
        for file_path, depth in import_depths.items():
            if file_path in import_status:
                import_status[file_path]["import_depth"] = depth
        
        # Find most/least imported files
        most_imported = max(import_status.items(), 
                          key=lambda x: x[1].get("import_count", 0)) if import_status else None
        least_imported = min(import_status.items(), 
                           key=lambda x: x[1].get("import_count", 0)) if import_status else None
        
        return {
            "import_status": import_status,
            "summary": {
                "total_files": total_files,
                "imported_files": imported_files,
                "unimported_files": unimported_files,
                "only_non_production_files": only_non_production_files,
                "import_percentage": (imported_files / total_files * 100) if total_files > 0 else 0,
                "non_production_percentage": (only_non_production_files / total_files * 100) if total_files > 0 else 0,
                "most_imported_file": {
                    "file": most_imported[0] if most_imported else None,
                    "import_count": most_imported[1].get("import_count", 0) if most_imported else 0
                },
                "least_imported_file": {
                    "file": least_imported[0] if least_imported else None,
                    "import_count": least_imported[1].get("import_count", 0) if least_imported else 0
                }
            },
            "advanced_analysis": {
                "circular_imports": circular_imports,
                "import_depths": import_depths,
                "critical_paths": critical_paths
            },
            "stats": self.stats
        }
    
    def _find_repository_root(self, directory_path: str) -> str:
        """Find the repository root by looking for common indicators."""
        current_path = Path(directory_path).resolve()
        
        # Look for .git directory first (most reliable indicator)
        while current_path != current_path.parent:
            if (current_path / '.git').exists():
                return str(current_path)
            current_path = current_path.parent
        
        # If no .git found, look for other indicators
        current_path = Path(directory_path).resolve()
        indicators = ['pyproject.toml', 'setup.py', 'README.md']
        
        while current_path != current_path.parent:
            for indicator in indicators:
                if (current_path / indicator).exists():
                    return str(current_path)
            current_path = current_path.parent
        
        # If no indicators found, use the original directory
        return directory_path
    
    def _get_module_name(self, file_path: Path, root_path: str) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(Path(root_path))
            module_parts = list(relative_path.parts)
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]
            return '.'.join(module_parts)
        except Exception:
            return str(file_path.stem)  # Fallback to filename without extension
    
    def _extract_all_imports(self, tree: ast.AST, file_path: Path, root_path: str) -> Set[str]:
        """Extract all imports from AST, including relative imports."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle relative imports
                    if node.level > 0:
                        # This is a relative import
                        current_module = self._get_module_name(file_path, root_path)
                        base_parts = current_module.split('.')
                        # Remove levels from the end
                        if node.level <= len(base_parts):
                            base_parts = base_parts[:-node.level]
                        if base_parts and node.module:
                            relative_module = '.'.join(base_parts + [node.module])
                            imports.add(relative_module)
                        elif base_parts:
                            relative_module = '.'.join(base_parts)
                            imports.add(relative_module)
                    else:
                        imports.add(node.module)
                
                # Also track specific imports from the module
                for alias in node.names:
                    if alias.name != '*':  # Skip wildcard imports
                        if node.module:
                            if node.level > 0:
                                # Handle relative imports for specific items
                                current_module = self._get_module_name(file_path, root_path)
                                base_parts = current_module.split('.')
                                if node.level <= len(base_parts):
                                    base_parts = base_parts[:-node.level]
                                if base_parts:
                                    full_name = '.'.join(base_parts + [node.module, alias.name])
                                    imports.add(full_name)
                            else:
                                full_name = f"{node.module}.{alias.name}"
                                imports.add(full_name)
                        else:
                            imports.add(alias.name)
        
        return imports
    
    def _is_module_imported(self, module_name: str, imports: Set[str]) -> bool:
        """Check if a module is imported by checking various import patterns."""
        # Direct module import
        if module_name in imports:
            return True
        
        # Check if any import is a submodule of this module (import starts with module_name + '.')
        # This handles cases like importing 'src.utils.vif_calculator' when looking for 'src.utils'
        for imp in imports:
            if imp.startswith(module_name + '.'):
                return True
        
        # REMOVED: Check if this module is a submodule of any import
        # This was causing false positives - importing 'src.utils' doesn't mean
        # 'src.utils.vif_calculator' is imported
        
        # Check for exact matches with different representations
        module_parts = module_name.split('.')
        for imp in imports:
            imp_parts = imp.split('.')
            # Check if the last part of the import matches the last part of the module
            if imp_parts and module_parts and imp_parts[-1] == module_parts[-1]:
                # Check if the import is a prefix of the module or vice versa
                if (len(imp_parts) <= len(module_parts) and 
                    module_parts[:len(imp_parts)] == imp_parts):
                    return True
                if (len(module_parts) <= len(imp_parts) and 
                    imp_parts[:len(module_parts)] == module_parts):
                    return True
        
        return False
    
    def _is_only_imported_by_non_production(self, imported_by: List[str]) -> bool:
        """Check if file is only imported by non-production files (fix, example, test, .md)."""
        if not imported_by:
            return False
        
        non_production_patterns = ["fix", "example", "test", ".md"]
        
        for importer in imported_by:
            importer_lower = importer.lower()
            is_non_production = any(pattern in importer_lower for pattern in non_production_patterns)
            if not is_non_production:
                return False  # Found at least one production import
        
        return True  # All imports are from non-production files
    
    def _detect_circular_imports(self, import_status: Dict[str, Any], file_to_module: Dict[str, str]) -> List[List[str]]:
        """Detect circular imports between files."""
        circular_imports = []
        visited = set()
        rec_stack = set()
        
        def dfs(file_path: str, path: List[str]) -> None:
            if file_path in rec_stack:
                # Found a cycle
                cycle_start = path.index(file_path)
                cycle = path[cycle_start:] + [file_path]
                circular_imports.append(cycle)
                return
            
            if file_path in visited:
                return
            
            visited.add(file_path)
            rec_stack.add(file_path)
            path.append(file_path)
            
            # Check all files that import this file
            file_info = import_status.get(file_path, {})
            imported_by = file_info.get("imported_by", [])
            
            for importer in imported_by:
                dfs(importer, path.copy())
            
            rec_stack.remove(file_path)
            path.pop()
        
        for file_path in import_status:
            if file_path not in visited:
                dfs(file_path, [])
        
        return circular_imports
    
    def _calculate_import_depths(self, import_status: Dict[str, Any], file_to_module: Dict[str, str]) -> Dict[str, int]:
        """Calculate the maximum import depth for each file."""
        depths = {}
        visited = set()
        
        def calculate_depth(file_path: str, current_path: List[str]) -> int:
            if file_path in current_path:
                # Circular dependency - return current depth
                return len(current_path)
            
            if file_path in depths:
                return depths[file_path]
            
            if file_path not in import_status:
                return 0
            
            current_path.append(file_path)
            max_depth = 0
            
            file_info = import_status[file_path]
            imported_by = file_info.get("imported_by", [])
            
            for importer in imported_by:
                depth = calculate_depth(importer, current_path.copy())
                max_depth = max(max_depth, depth)
            
            current_path.pop()
            depth = max_depth + 1
            depths[file_path] = depth
            return depth
        
        for file_path in import_status:
            if file_path not in visited:
                calculate_depth(file_path, [])
                visited.add(file_path)
        
        return depths
    
    def _identify_critical_paths(self, import_status: Dict[str, Any]) -> Dict[str, Any]:
        """Identify critical files that, if removed, would break many others."""
        critical_paths = {
            "high_impact_files": [],
            "dependency_chains": {},
            "bottleneck_files": []
        }
        
        # Find files with high import counts (many depend on them)
        import_counts = {}
        for file_path, file_info in import_status.items():
            import_count = file_info.get("import_count", 0)
            import_counts[file_path] = import_count
        
        # Sort by import count to find high-impact files
        sorted_files = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Top 10% of files by import count are considered high-impact
        top_count = max(1, len(sorted_files) // 10)
        critical_paths["high_impact_files"] = sorted_files[:top_count]
        
        # Find bottleneck files (files that import many others)
        bottleneck_files = []
        for file_path, file_info in import_status.items():
            # This would require analyzing what each file imports, not just what imports it
            # For now, we'll use a simple heuristic based on import count
            if file_info.get("import_count", 0) > 5:  # Arbitrary threshold
                bottleneck_files.append((file_path, file_info.get("import_count", 0)))
        
        critical_paths["bottleneck_files"] = sorted(bottleneck_files, key=lambda x: x[1], reverse=True)
        
        return critical_paths
    
    def get_simple_yes_no_report(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a simple yes/no report for each file.
        
        Returns:
            Dict mapping file paths to "YES" or "NO" based on import status
        """
        simple_report = {}
        import_status = results.get("import_status", {})
        
        for file_path, status in import_status.items():
            simple_report[file_path] = "YES" if status["is_imported"] else "NO"
        
        return simple_report
    
    def print_simple_report(self, results: Dict[str, Any]) -> None:
        """Print a simple yes/no report to console with import details."""
        simple_report = self.get_simple_yes_no_report(results)
        summary = results.get("summary", {})
        import_status = results.get("import_status", {})
        
        tprint("\n" + "="*80)
        tprint("IMPORT VERIFICATION REPORT")
        tprint("="*80)
        tprint(f"Total files analyzed: {summary.get('total_files', 0)}")
        tprint(f"Files imported by others: {summary.get('imported_files', 0)}")
        tprint(f"Files NOT imported by others: {summary.get('unimported_files', 0)}")
        tprint(f"Files only imported by non-production files: {summary.get('only_non_production_files', 0)}")
        tprint(f"Import percentage: {summary.get('import_percentage', 0):.1f}%")
        tprint(f"Non-production only percentage: {summary.get('non_production_percentage', 0):.1f}%")
        tprint("\n" + "-"*80)
        tprint("FILE IMPORT STATUS (YES = imported by others, NO = not imported)")
        tprint("-"*80)
        
        # Sort files for consistent output
        for file_path in sorted(simple_report.keys()):
            status = simple_report[file_path]
            file_info = import_status.get(file_path, {})
            imported_by = file_info.get("imported_by", [])
            only_non_production = file_info.get("only_imported_by_non_production", False)
            import_depth = file_info.get("import_depth", 0)
            
            # Show relative path for better readability
            try:
                rel_path = Path(file_path).relative_to(Path.cwd())
                # Add flag indicators
                flag_indicator = " [NON-PROD]" if only_non_production else ""
                depth_indicator = f" [D{import_depth}]" if import_depth > 0 else ""
                tprint(f"{status:3} | {rel_path}{flag_indicator}{depth_indicator}")
                
                # Show which files import this file (if any)
                if imported_by:
                    tprint(f"     └─ Imported by {len(imported_by)} file(s):")
                    for importer in sorted(imported_by):
                        try:
                            rel_importer = Path(importer).relative_to(Path.cwd())
                            tprint(f"        • {rel_importer}")
                        except ValueError:
                            tprint(f"        • {importer}")
                elif status == "NO":
                    tprint(f"     └─ Not imported by any other files")
                    
            except ValueError:
                flag_indicator = " [NON-PROD]" if only_non_production else ""
                depth_indicator = f" [D{import_depth}]" if import_depth > 0 else ""
                tprint(f"{status:3} | {file_path}{flag_indicator}{depth_indicator}")
                if imported_by:
                    tprint(f"     └─ Imported by {len(imported_by)} file(s):")
                    for importer in sorted(imported_by):
                        tprint(f"        • {importer}")
                elif status == "NO":
                    tprint(f"     └─ Not imported by any other files")
        
        tprint("\n" + "-"*80)
        most_imported = summary.get("most_imported_file", {})
        if most_imported.get("file"):
            tprint(f"Most imported file: {most_imported['file']} ({most_imported['import_count']} imports)")
        
        least_imported = summary.get("least_imported_file", {})
        if least_imported.get("file"):
            tprint(f"Least imported file: {least_imported['file']} ({least_imported['import_count']} imports)")
        
        # Advanced analysis section
        self._print_advanced_analysis(results)
    
    def _print_advanced_analysis(self, results: Dict[str, Any]) -> None:
        """Print advanced analysis results."""
        advanced = results.get("advanced_analysis", {})
        
        tprint("\n" + "="*80)
        tprint("ADVANCED ANALYSIS")
        tprint("="*80)
        
        # Circular imports
        circular_imports = advanced.get("circular_imports", [])
        tprint(f"\n🔄 CIRCULAR IMPORTS: {len(circular_imports)} found")
        if circular_imports:
            for i, cycle in enumerate(circular_imports, 1):
                tprint(f"  {i}. {' → '.join(cycle)}")
        else:
            tprint("  ✅ No circular imports detected")
        
        # Import depths
        import_depths = advanced.get("import_depths", {})
        if import_depths:
            max_depth = max(import_depths.values()) if import_depths else 0
            avg_depth = sum(import_depths.values()) / len(import_depths) if import_depths else 0
            tprint(f"\n📊 IMPORT DEPTH ANALYSIS:")
            tprint(f"  Maximum depth: {max_depth}")
            tprint(f"  Average depth: {avg_depth:.1f}")
            
            # Show files with highest depths
            sorted_depths = sorted(import_depths.items(), key=lambda x: x[1], reverse=True)
            tprint(f"  Deepest import chains:")
            for file_path, depth in sorted_depths[:5]:
                try:
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    tprint(f"    • {rel_path} (depth: {depth})")
                except ValueError:
                    tprint(f"    • {file_path} (depth: {depth})")
        
        # Critical paths
        critical_paths = advanced.get("critical_paths", {})
        high_impact_files = critical_paths.get("high_impact_files", [])
        bottleneck_files = critical_paths.get("bottleneck_files", [])
        
        tprint(f"\n🎯 CRITICAL PATH ANALYSIS:")
        tprint(f"  High-impact files (top 10% by import count): {len(high_impact_files)}")
        if high_impact_files:
            for file_path, count in high_impact_files[:5]:
                try:
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    tprint(f"    • {rel_path} ({count} imports)")
                except ValueError:
                    tprint(f"    • {file_path} ({count} imports)")
        
        tprint(f"  Bottleneck files (high import count): {len(bottleneck_files)}")
        if bottleneck_files:
            for file_path, count in bottleneck_files[:5]:
                try:
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    tprint(f"    • {rel_path} ({count} imports)")
                except ValueError:
                    tprint(f"    • {file_path} ({count} imports)")