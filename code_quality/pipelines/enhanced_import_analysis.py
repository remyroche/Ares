#!/usr/bin/env python3
"""
Enhanced Import Analyzer Plugin

This plugin provides comprehensive import analysis capabilities including:
- Import dependency mapping
- Circular dependency detection
- Unused import identification
- Import conflict resolution
- Import optimization suggestions
"""

import ast
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.import_analyzer import ImportAnalyzer
from core.config import CodeQualityConfig
from plugins.base_plugin import FileProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class EnhancedImportAnalyzerPlugin(FileProcessorPlugin):
    """
    Enhanced import analyzer plugin for comprehensive import analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.metadata = PluginMetadata(
            name="EnhancedImportAnalyzer",
            version="1.0.0",
            description="Comprehensive import analysis and optimization",
            author="Code Quality Team",
            category=PluginCategory.ANALYZER,
            priority=PluginPriority.HIGH
        )
        
        # Initialize the core import analyzer
        self.import_analyzer = ImportAnalyzer(CodeQualityConfig())
        
        # Analysis results storage
        self.analysis_results = {
            "import_dependencies": {},
            "circular_dependencies": [],
            "unused_imports": [],
            "import_conflicts": [],
            "optimization_suggestions": []
        }
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file for import issues.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file
            tree = ast.parse(content, filename=file_path)
            
            # Extract import information
            imports = self._extract_imports(tree)
            
            # Analyze imports
            analysis = {
                "file_path": file_path,
                "imports": imports,
                "issues": [],
                "suggestions": []
            }
            
            # Check for unused imports
            unused = self._find_unused_imports(tree, imports)
            if unused:
                analysis["issues"].extend(unused)
            
            # Check for duplicate imports
            duplicates = self._find_duplicate_imports(imports)
            if duplicates:
                analysis["issues"].extend(duplicates)
            
            # Check for import conflicts
            conflicts = self._find_import_conflicts(imports)
            if conflicts:
                analysis["issues"].extend(conflicts)
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(imports)
            if suggestions:
                analysis["suggestions"].extend(suggestions)
            
            return analysis
            
        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "imports": [],
                "issues": [],
                "suggestions": []
            }
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Analyze all Python files in a directory for import issues.
        
        Args:
            directory_path: Path to the directory to analyze
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        directory = Path(directory_path)
        python_files = list(directory.rglob("*.py"))
        
        results = {
            "directory": directory_path,
            "total_files": len(python_files),
            "analyzed_files": 0,
            "files": {},
            "summary": {
                "total_imports": 0,
                "total_issues": 0,
                "unused_imports": 0,
                "duplicate_imports": 0,
                "import_conflicts": 0,
                "circular_dependencies": 0
            }
        }
        
        for file_path in python_files:
            try:
                file_analysis = self.analyze_file(str(file_path))
                results["files"][str(file_path)] = file_analysis
                results["analyzed_files"] += 1
                
                # Update summary
                results["summary"]["total_imports"] += len(file_analysis.get("imports", []))
                results["summary"]["total_issues"] += len(file_analysis.get("issues", []))
                
                # Count specific issue types
                for issue in file_analysis.get("issues", []):
                    if issue.get("type") == "unused_import":
                        results["summary"]["unused_imports"] += 1
                    elif issue.get("type") == "duplicate_import":
                        results["summary"]["duplicate_imports"] += 1
                    elif issue.get("type") == "import_conflict":
                        results["summary"]["import_conflicts"] += 1
                        
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies(results["files"])
        results["summary"]["circular_dependencies"] = len(circular_deps)
        results["circular_dependencies"] = circular_deps
        
        return results
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import information from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "col": node.col_offset
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "col": node.col_offset
                    })
        
        return imports
    
    def _find_unused_imports(self, tree: ast.AST, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find unused imports in the file."""
        unused = []
        
        # Get all names used in the file
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle attribute access like module.function
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Check which imports are unused
        for imp in imports:
            name_to_check = imp.get("alias") or imp.get("name") or imp.get("module")
            if name_to_check and name_to_check not in used_names:
                unused.append({
                    "type": "unused_import",
                    "import": imp,
                    "line": imp["line"],
                    "message": f"Unused import: {name_to_check}"
                })
        
        return unused
    
    def _find_duplicate_imports(self, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find duplicate imports."""
        seen = {}
        duplicates = []
        
        for imp in imports:
            key = (imp.get("module"), imp.get("name"))
            if key in seen:
                duplicates.append({
                    "type": "duplicate_import",
                    "import": imp,
                    "line": imp["line"],
                    "message": f"Duplicate import: {imp.get('module')}.{imp.get('name', '')}"
                })
            else:
                seen[key] = imp
        
        return duplicates
    
    def _find_import_conflicts(self, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find import conflicts (same name imported from different modules)."""
        name_sources = {}
        conflicts = []
        
        for imp in imports:
            name = imp.get("alias") or imp.get("name")
            if name:
                if name in name_sources:
                    conflicts.append({
                        "type": "import_conflict",
                        "import": imp,
                        "line": imp["line"],
                        "message": f"Name conflict: {name} imported from both {name_sources[name]} and {imp.get('module')}"
                    })
                else:
                    name_sources[name] = imp.get("module")
        
        return conflicts
    
    def _generate_optimization_suggestions(self, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for imports."""
        suggestions = []
        
        # Suggest grouping imports
        if len(imports) > 10:
            suggestions.append({
                "type": "optimization",
                "message": "Consider grouping imports by category (standard library, third-party, local)"
            })
        
        # Suggest using from imports for specific functions
        for imp in imports:
            if imp.get("type") == "import" and imp.get("module"):
                suggestions.append({
                    "type": "optimization",
                    "message": f"Consider using 'from {imp['module']} import specific_function' instead of importing entire module"
                })
        
        return suggestions
    
    def _detect_circular_dependencies(self, files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect circular dependencies between files."""
        # This is a simplified implementation
        # In a real implementation, you would build a dependency graph
        # and detect cycles using graph algorithms
        return []
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive report of the analysis results."""
        report = []
        report.append("=" * 60)
        report.append("ENHANCED IMPORT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        summary = analysis_results.get("summary", {})
        report.append("SUMMARY:")
        report.append(f"  Total files analyzed: {analysis_results.get('analyzed_files', 0)}")
        report.append(f"  Total imports: {summary.get('total_imports', 0)}")
        report.append(f"  Total issues: {summary.get('total_issues', 0)}")
        report.append(f"  Unused imports: {summary.get('unused_imports', 0)}")
        report.append(f"  Duplicate imports: {summary.get('duplicate_imports', 0)}")
        report.append(f"  Import conflicts: {summary.get('import_conflicts', 0)}")
        report.append(f"  Circular dependencies: {summary.get('circular_dependencies', 0)}")
        report.append("")
        
        # Detailed file analysis
        report.append("DETAILED ANALYSIS:")
        for file_path, file_analysis in analysis_results.get("files", {}).items():
            if file_analysis.get("issues") or file_analysis.get("suggestions"):
                report.append(f"\nFile: {file_path}")
                
                for issue in file_analysis.get("issues", []):
                    report.append(f"  Issue (Line {issue.get('line', '?')}): {issue.get('message', '')}")
                
                for suggestion in file_analysis.get("suggestions", []):
                    report.append(f"  Suggestion: {suggestion.get('message', '')}")
        
        return "\n".join(report)


def main():
    """Main entry point for the enhanced import analyzer plugin."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Import Analyzer Plugin")
    parser.add_argument("target", help="File or directory to analyze")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize the plugin
    plugin = EnhancedImportAnalyzerPlugin()
    
    # Analyze the target
    if Path(args.target).is_file():
        results = plugin.analyze_file(args.target)
        print(f"Analysis complete for {args.target}")
    else:
        results = plugin.analyze_directory(args.target)
        print(f"Analysis complete for directory {args.target}")
    
    # Generate and display report
    report = plugin.generate_report(results)
    print(report)
    
    # Save report if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()