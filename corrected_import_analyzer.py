#!/usr/bin/env python3
"""
Corrected Import Analyzer - Fixes the fundamental bug in import detection.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


class CorrectedImportAnalyzer:
    """Corrected analyzer that properly detects all import patterns."""
    
    def __init__(self, project_root: str = "/workspace"):
        self.project_root = Path(project_root)
        self.stats = {
            "files_analyzed": 0,
            "total_imports": 0,
            "errors": 0
        }
    
    def analyze_utils_directory(self) -> Dict[str, Any]:
        """Analyze the src/utils/ directory for dead code."""
        utils_dir = self.project_root / "src" / "utils"
        
        if not utils_dir.exists():
            return {"error": f"Directory {utils_dir} does not exist"}
        
        # Get all Python files in utils directory
        utils_files = list(utils_dir.rglob("*.py"))
        utils_files = [f for f in utils_files if f.name != "__init__.py"]
        
        # Get all Python files in the entire project
        all_python_files = list(self.project_root.rglob("*.py"))
        
        # Build module mapping
        module_to_file = {}
        file_to_module = {}
        
        for file_path in all_python_files:
            module_name = self._get_module_name(file_path)
            module_to_file[module_name] = file_path
            file_to_module[str(file_path)] = module_name
        
        # Collect all imports from all files
        all_imports = {}
        for file_path in all_python_files:
            imports = self._extract_all_imports(file_path)
            all_imports[str(file_path)] = imports
            self.stats["files_analyzed"] += 1
            self.stats["total_imports"] += len(imports)
        
        # Check which utils files are imported
        import_status = {}
        for utils_file in utils_files:
            file_path_str = str(utils_file)
            module_name = file_to_module.get(file_path_str)
            
            if not module_name:
                import_status[file_path_str] = {
                    "is_imported": False,
                    "imported_by": [],
                    "reason": "Could not determine module name"
                }
                continue
            
            # Check if this module is imported by any other file
            imported_by = []
            for other_file, other_imports in all_imports.items():
                if other_file != file_path_str:
                    if self._is_module_imported_corrected(module_name, other_imports):
                        imported_by.append(other_file)
            
            is_imported = len(imported_by) > 0
            
            import_status[file_path_str] = {
                "is_imported": is_imported,
                "imported_by": imported_by,
                "import_count": len(imported_by),
                "module_name": module_name
            }
        
        # Calculate summary
        total_files = len(utils_files)
        imported_files = sum(1 for status in import_status.values() if status["is_imported"])
        unimported_files = total_files - imported_files
        
        return {
            "import_status": import_status,
            "summary": {
                "total_files": total_files,
                "imported_files": imported_files,
                "unimported_files": unimported_files,
                "import_percentage": (imported_files / total_files * 100) if total_files > 0 else 0
            },
            "stats": self.stats
        }
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(self.project_root)
            module_parts = list(relative_path.parts)
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]
            return '.'.join(module_parts)
        except Exception:
            return str(file_path.stem)
    
    def _extract_all_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            self.stats["errors"] += 1
            return imports
        
        try:
            tree = ast.parse(content)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            self.stats["errors"] += 1
            return imports
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Add the module itself
                    imports.add(node.module)
                    
                    # Add specific imports from the module
                    for alias in node.names:
                        if alias.name != '*':
                            full_import = f"{node.module}.{alias.name}"
                            imports.add(full_import)
        
        return imports
    
    def _is_module_imported_corrected(self, module_name: str, imports: Set[str]) -> bool:
        """
        CORRECTED: Check if a module is imported with proper logic.
        The key fix: if ANY import starts with the module name, the module is imported.
        """
        # Direct exact match
        if module_name in imports:
            return True
        
        # CORRECTED LOGIC: Check if any import starts with this module name
        # This catches cases like:
        # - module_name = "src.utils.sr_clustering"
        # - imports contains "src.utils.sr_clustering.get_backtesting_enhanced_clustering"
        for imp in imports:
            if imp.startswith(module_name + '.'):
                return True
        
        return False
    
    def print_detailed_report(self, results: Dict[str, Any]) -> None:
        """Print a detailed report with import examples."""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        summary = results.get("summary", {})
        import_status = results.get("import_status", {})
        
        print("\n" + "="*80)
        print("CORRECTED DEAD CODE ANALYSIS REPORT - src/utils/")
        print("="*80)
        print(f"Total files analyzed: {summary.get('total_files', 0)}")
        print(f"Files imported by others: {summary.get('imported_files', 0)}")
        print(f"Files NOT imported by others (potential dead code): {summary.get('unimported_files', 0)}")
        print(f"Import percentage: {summary.get('import_percentage', 0):.1f}%")
        print("\n" + "-"*80)
        print("DETAILED RESULTS:")
        print("-"*80)
        
        # Sort files for consistent output
        for file_path in sorted(import_status.keys()):
            status = import_status[file_path]
            is_imported = status["is_imported"]
            imported_by = status["imported_by"]
            import_count = status["import_count"]
            module_name = status.get("module_name", "unknown")
            
            # Show relative path
            try:
                rel_path = Path(file_path).relative_to(Path.cwd())
                status_text = "IMPORTED" if is_imported else "DEAD CODE"
                print(f"{status_text:10} | {rel_path}")
                print(f"             Module: {module_name} ({import_count} imports)")
                
                if imported_by:
                    print(f"             └─ Imported by:")
                    for importer in sorted(imported_by)[:3]:  # Show first 3
                        try:
                            rel_importer = Path(importer).relative_to(Path.cwd())
                            print(f"                • {rel_importer}")
                        except ValueError:
                            print(f"                • {importer}")
                    if len(imported_by) > 3:
                        print(f"                ... and {len(imported_by) - 3} more")
                elif not is_imported:
                    print(f"             └─ Not imported by any other files")
                print()
                    
            except ValueError:
                status_text = "IMPORTED" if is_imported else "DEAD CODE"
                print(f"{status_text:10} | {file_path}")
                print(f"             Module: {module_name} ({import_count} imports)")
                if imported_by:
                    print(f"             └─ Imported by:")
                    for importer in sorted(imported_by)[:3]:
                        print(f"                • {importer}")
                    if len(imported_by) > 3:
                        print(f"                ... and {len(imported_by) - 3} more")
                elif not is_imported:
                    print(f"             └─ Not imported by any other files")
                print()
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        
        # List dead code files
        dead_code_files = [f for f, status in import_status.items() if not status["is_imported"]]
        if dead_code_files:
            print(f"\n🚨 POTENTIAL DEAD CODE FILES ({len(dead_code_files)}):")
            for file_path in sorted(dead_code_files):
                try:
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    module_name = import_status[file_path].get("module_name", "unknown")
                    print(f"  • {rel_path} (module: {module_name})")
                except ValueError:
                    print(f"  • {file_path}")
        else:
            print("\n✅ No dead code found - all files are imported by others")
        
        # List most imported files
        most_imported = sorted(import_status.items(), key=lambda x: x[1]["import_count"], reverse=True)
        if most_imported:
            print(f"\n📈 MOST IMPORTED FILES:")
            for file_path, status in most_imported[:5]:
                if status["import_count"] > 0:
                    try:
                        rel_path = Path(file_path).relative_to(Path.cwd())
                        module_name = status.get("module_name", "unknown")
                        print(f"  • {rel_path} ({status['import_count']} imports, module: {module_name})")
                    except ValueError:
                        print(f"  • {file_path} ({status['import_count']} imports)")


def main():
    """Main function to run the analysis."""
    analyzer = CorrectedImportAnalyzer()
    results = analyzer.analyze_utils_directory()
    analyzer.print_detailed_report(results)
    
    # Return exit code based on results
    if "error" in results:
        return 1
    
    summary = results.get("summary", {})
    unimported_files = summary.get("unimported_files", 0)
    
    if unimported_files > 0:
        print(f"\n⚠️  Found {unimported_files} potential dead code files")
        return 1
    else:
        print(f"\n✅ No dead code found")
        return 0


if __name__ == "__main__":
    exit(main())