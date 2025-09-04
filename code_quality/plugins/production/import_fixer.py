"""
Production Import Fixer Plugin

A robust, production-ready plugin for fixing Python import issues with
comprehensive analysis, intelligent fixing strategies, and detailed reporting.
"""

import ast
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from code_quality.plugins.base_plugin import FileProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class ProductionImportFixerPlugin(FileProcessorPlugin):
    """
    Production-ready plugin for fixing Python import issues.
    
    Features:
    - Comprehensive import analysis and fixing
    - Unused import detection and removal
    - Duplicate import detection and consolidation
    - Import sorting and organization
    - Circular import detection
    - Backup creation and rollback capabilities
    - Detailed reporting and metrics
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="production_import_fixer",
            version="2.0.0",
            description="Production-ready Python import fixer with comprehensive analysis and intelligent fixing",
            author="Code Quality Team",
            category=PluginCategory.IMPORT,
            priority=PluginPriority.HIGH,
            dependencies=[],
            tags={"imports", "fixing", "python", "production", "analysis"},
            required_packages=[],
            optional_packages=["isort", "autoflake"],
            configuration_schema={
                "create_backups": {"type": "boolean", "default": True},
                "backup_suffix": {"type": "string", "default": ".bak"},
                "remove_unused": {"type": "boolean", "default": True},
                "remove_duplicates": {"type": "boolean", "default": True},
                "sort_imports": {"type": "boolean", "default": True},
                "detect_circular": {"type": "boolean", "default": True},
                "group_imports": {"type": "boolean", "default": True},
                "import_order": {"type": "string", "default": "isort"},
                "max_line_length": {"type": "integer", "default": 120},
                "preserve_imports": {"type": "list", "default": []},
                "aggressive_removal": {"type": "boolean", "default": False},
                "timeout_seconds": {"type": "integer", "default": 30}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        return True  # Only requires standard library
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi', '.pyw'}
    
    def process_file(self, file_path: Path, context) -> Dict[str, Any]:
        """
        Process a single Python file for import issues with comprehensive analysis.
        
        Args:
            file_path: Path to the Python file
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Detailed processing result
        """
        result = {
            "success": True,
            "issues_found": 0,
            "issues_fixed": 0,
            "error": None,
            "warnings": [],
            "fixes_applied": [],
            "backup_created": False,
            "processing_time": 0.0,
            "import_analysis": {}
        }
        
        start_time = datetime.now()
        
        try:
            # Validate file
            if not self._validate_file(file_path):
                result["success"] = False
                result["error"] = "File validation failed"
                return result
            
            # Create backup if configured
            backup_path = None
            if self.configuration.get("create_backups", True) and not context.dry_run:
                backup_path = self._create_backup(file_path)
                if backup_path:
                    result["backup_created"] = True
                    result["backup_path"] = str(backup_path)
            
            # Read file content
            content = self._read_file(file_path)
            if content is None:
                result["success"] = False
                result["error"] = "Failed to read file"
                return result
            
            # Parse and analyze imports
            try:
                tree = ast.parse(content)
            except SyntaxError:
                result["success"] = False
                result["error"] = "File has syntax errors, skipping import fixes"
                return result
            
            # Analyze imports
            import_analysis = self._analyze_imports(tree, content, file_path)
            result["import_analysis"] = import_analysis
            
            # Count issues
            issues = []
            if self.configuration.get("remove_unused", True):
                issues.extend(import_analysis.get("unused_imports", []))
            if self.configuration.get("remove_duplicates", True):
                issues.extend(import_analysis.get("duplicate_imports", []))
            if self.configuration.get("sort_imports", True):
                issues.extend(import_analysis.get("unsorted_imports", []))
            
            result["issues_found"] = len(issues)
            
            if issues and not context.dry_run:
                # Fix import issues
                fixed_content, fixes_applied = self._fix_import_issues(content, import_analysis)
                
                if fixed_content != content:
                    # Write fixed content
                    success = self._write_file(file_path, fixed_content)
                    if success:
                        result["issues_fixed"] = len(fixes_applied)
                        result["fixes_applied"] = fixes_applied
                        result["success"] = True
                    else:
                        result["success"] = False
                        result["error"] = "Failed to write fixed content"
                        # Restore backup if available
                        if backup_path and backup_path.exists():
                            shutil.copy2(backup_path, file_path)
                else:
                    result["success"] = False
                    result["error"] = "Could not fix import issues"
                    result["warnings"].append("No fixes could be applied")
            elif issues:
                result["success"] = False
                result["error"] = f"Found {len(issues)} import issues (dry run)"
                result["warnings"].append("Dry run mode - no fixes applied")
            else:
                result["success"] = True
                result["warnings"].append("No import issues found")
        
        except Exception as e:
            result["success"] = False
            result["error"] = f"Unexpected error: {str(e)}"
            result["warnings"].append(f"Exception during processing: {type(e).__name__}")
        
        finally:
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate that the file can be processed."""
        try:
            if not file_path.exists():
                return False
            
            if not file_path.is_file():
                return False
            
            if file_path.stat().st_size == 0:
                return False
            
            # Check file size (prevent processing huge files)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_path.stat().st_size > max_size:
                return False
            
            return True
        except Exception:
            return False
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of the file."""
        try:
            backup_suffix = self.configuration.get("backup_suffix", ".bak")
            backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
            
            # Ensure backup path is unique
            counter = 1
            while backup_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.{counter}{backup_suffix}")
                counter += 1
            
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception:
            return None
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def _write_file(self, file_path: Path, content: str) -> bool:
        """Write file content."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def _analyze_imports(self, tree: ast.AST, content: str, file_path: Path) -> Dict[str, Any]:
        """Analyze imports in the AST and identify issues."""
        analysis = {
            "imports": [],
            "unused_imports": [],
            "duplicate_imports": [],
            "unsorted_imports": [],
            "circular_imports": [],
            "used_names": set(),
            "import_lines": []
        }
        
        # Collect all imports and used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = {
                        "type": "import",
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "column": node.col_offset,
                        "full_name": alias.name
                    }
                    analysis["imports"].append(import_info)
                    analysis["import_lines"].append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_info = {
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "column": node.col_offset,
                        "full_name": f"{module}.{alias.name}" if module else alias.name
                    }
                    analysis["imports"].append(import_info)
                    analysis["import_lines"].append(node.lineno)
            elif isinstance(node, ast.Name):
                analysis["used_names"].add(node.id)
        
        # Find unused imports
        if self.configuration.get("remove_unused", True):
            analysis["unused_imports"] = self._find_unused_imports(analysis["imports"], analysis["used_names"])
        
        # Find duplicate imports
        if self.configuration.get("remove_duplicates", True):
            analysis["duplicate_imports"] = self._find_duplicate_imports(analysis["imports"])
        
        # Check import sorting
        if self.configuration.get("sort_imports", True):
            analysis["unsorted_imports"] = self._check_import_sorting(analysis["imports"])
        
        # Detect circular imports (basic check)
        if self.configuration.get("detect_circular", True):
            analysis["circular_imports"] = self._detect_circular_imports(analysis["imports"], file_path)
        
        return analysis
    
    def _find_unused_imports(self, imports: List[Dict[str, Any]], used_names: Set[str]) -> List[Dict[str, Any]]:
        """Find unused imports."""
        unused = []
        preserve_imports = set(self.configuration.get("preserve_imports", []))
        
        for imp in imports:
            if imp["type"] == "import":
                name = imp["alias"] or imp["name"].split('.')[0]
                if name not in used_names and imp["name"] not in preserve_imports:
                    unused.append(imp)
            elif imp["type"] == "from_import":
                name = imp["alias"] or imp["name"]
                if name not in used_names and imp["full_name"] not in preserve_imports:
                    unused.append(imp)
        
        return unused
    
    def _find_duplicate_imports(self, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find duplicate imports."""
        seen = {}
        duplicates = []
        
        for imp in imports:
            key = (imp["type"], imp.get("module", ""), imp["name"])
            if key in seen:
                duplicates.append(imp)
            else:
                seen[key] = imp
        
        return duplicates
    
    def _check_import_sorting(self, imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check if imports are properly sorted."""
        unsorted = []
        
        # Group imports by type
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for imp in imports:
            if self._is_stdlib_import(imp):
                stdlib_imports.append(imp)
            elif self._is_local_import(imp):
                local_imports.append(imp)
            else:
                third_party_imports.append(imp)
        
        # Check if imports are in correct order
        all_imports = stdlib_imports + third_party_imports + local_imports
        for i, imp in enumerate(all_imports):
            if i > 0:
                prev_imp = all_imports[i-1]
                if not self._is_import_order_correct(prev_imp, imp):
                    unsorted.append(imp)
        
        return unsorted
    
    def _is_stdlib_import(self, imp: Dict[str, Any]) -> bool:
        """Check if import is from standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 'collections',
            'itertools', 'functools', 'operator', 're', 'math', 'random',
            'string', 'io', 'tempfile', 'shutil', 'glob', 'fnmatch'
        }
        
        if imp["type"] == "import":
            return imp["name"].split('.')[0] in stdlib_modules
        elif imp["type"] == "from_import":
            return imp["module"] in stdlib_modules
        
        return False
    
    def _is_local_import(self, imp: Dict[str, Any]) -> bool:
        """Check if import is local (relative)."""
        if imp["type"] == "from_import":
            return imp["module"].startswith('.')
        return False
    
    def _is_import_order_correct(self, prev_imp: Dict[str, Any], curr_imp: Dict[str, Any]) -> bool:
        """Check if import order is correct."""
        prev_is_stdlib = self._is_stdlib_import(prev_imp)
        prev_is_local = self._is_local_import(prev_imp)
        curr_is_stdlib = self._is_stdlib_import(curr_imp)
        curr_is_local = self._is_local_import(curr_imp)
        
        # Standard library should come first
        if curr_is_stdlib and not prev_is_stdlib:
            return False
        
        # Local imports should come last
        if prev_is_local and not curr_is_local:
            return False
        
        return True
    
    def _detect_circular_imports(self, imports: List[Dict[str, Any]], file_path: Path) -> List[Dict[str, Any]]:
        """Detect potential circular imports (basic implementation)."""
        circular = []
        
        # This is a simplified check - in a real implementation,
        # you would need to analyze the entire project structure
        current_module = file_path.stem
        
        for imp in imports:
            if imp["type"] == "from_import" and imp["module"]:
                if imp["module"] == current_module:
                    circular.append(imp)
        
        return circular
    
    def _fix_import_issues(self, content: str, analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Fix import issues in content."""
        lines = content.split('\n')
        fixes_applied = []
        
        # Remove unused imports
        if self.configuration.get("remove_unused", True):
            unused_imports = analysis.get("unused_imports", [])
            for imp in sorted(unused_imports, key=lambda x: x["line"], reverse=True):
                line_num = imp["line"] - 1
                if line_num < len(lines):
                    lines.pop(line_num)
                    fixes_applied.append(f"Removed unused import: {imp['name']}")
        
        # Remove duplicate imports
        if self.configuration.get("remove_duplicates", True):
            duplicate_imports = analysis.get("duplicate_imports", [])
            for imp in sorted(duplicate_imports, key=lambda x: x["line"], reverse=True):
                line_num = imp["line"] - 1
                if line_num < len(lines):
                    lines.pop(line_num)
                    fixes_applied.append(f"Removed duplicate import: {imp['name']}")
        
        # Sort imports
        if self.configuration.get("sort_imports", True):
            sorted_content = self._sort_imports('\n'.join(lines))
            if sorted_content != '\n'.join(lines):
                lines = sorted_content.split('\n')
                fixes_applied.append("Sorted imports")
        
        return '\n'.join(lines), fixes_applied
    
    def _sort_imports(self, content: str) -> str:
        """Sort imports in content."""
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        in_import_section = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith(('import ', 'from ')):
                if not in_import_section:
                    in_import_section = True
                import_lines.append((i, line))
            elif stripped == '' and in_import_section:
                # Empty line in import section
                import_lines.append((i, line))
            else:
                if in_import_section:
                    in_import_section = False
                other_lines.append((i, line))
        
        # Sort import lines
        import_lines.sort(key=lambda x: x[1])
        
        # Reconstruct content
        result_lines = [''] * len(lines)
        
        for i, line in import_lines:
            result_lines[i] = line
        
        for i, line in other_lines:
            result_lines[i] = line
        
        return '\n'.join(result_lines)
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Production Import Fixer: Processing {len(context.target_files)} files")
        print(f"Configuration: remove_unused={self.configuration.get('remove_unused', True)}, "
              f"sort_imports={self.configuration.get('sort_imports', True)}")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Production Import Fixer: Fixed {result.issues_fixed} import issues in {result.files_processed} files")
            if result.issues_fixed > 0:
                print(f"Fixes applied: {', '.join(result.fixes_applied[:3])}{'...' if len(result.fixes_applied) > 3 else ''}")
        else:
            print(f"Production Import Fixer: Failed to process some files")
            if result.error:
                print(f"Error: {result.error}")
        
        if result.warnings:
            for warning in result.warnings[:3]:  # Show first 3 warnings
                print(f"Warning: {warning}")