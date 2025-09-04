"""
Import Fixer Plugin Example

Demonstrates how to create a plugin for import fixing functionality.
"""

import ast
import re
from pathlib import Path
from typing import Dict, Any, Set, List
from code_quality.plugins.base_plugin import FileProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class ImportFixerPlugin(FileProcessorPlugin):
    """
    Plugin for fixing Python import issues.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="import_fixer",
            version="1.0.0",
            description="Fixes common Python import issues",
            author="Code Quality Team",
            category=PluginCategory.IMPORT,
            priority=PluginPriority.HIGH,
            dependencies=[],
            tags={"imports", "fixing", "python"},
            required_packages=[],
            optional_packages=["isort"],
            configuration_schema={
                "sort_imports": {"type": "boolean", "default": True},
                "remove_unused": {"type": "boolean", "default": True},
                "add_missing": {"type": "boolean", "default": True},
                "import_order": {"type": "string", "default": "isort"}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        return True
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi'}
    
    def process_file(self, file_path: Path, context) -> Dict[str, Any]:
        """
        Process a single Python file for import issues.
        
        Args:
            file_path: Path to the Python file
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Processing result
        """
        result = {
            "success": True,
            "issues_found": 0,
            "issues_fixed": 0,
            "error": None
        }
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # Skip files with syntax errors
                result["success"] = False
                result["error"] = "File has syntax errors, skipping import fixes"
                return result
            
            # Analyze imports
            import_issues = self._analyze_imports(tree, content)
            result["issues_found"] = len(import_issues)
            
            if import_issues and not context.dry_run:
                # Fix import issues
                fixed_content = self._fix_imports(content, import_issues)
                if fixed_content != content:
                    # Write fixed content back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    result["issues_fixed"] = len(import_issues)
                    result["success"] = True
                else:
                    result["success"] = False
                    result["error"] = "Could not fix import issues"
            elif import_issues:
                result["success"] = False
                result["error"] = f"Found {len(import_issues)} import issues (dry run)"
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _analyze_imports(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """
        Analyze imports in the AST and identify issues.
        
        Args:
            tree: Parsed AST
            content: Original file content
            
        Returns:
            List[Dict[str, Any]]: List of import issues found
        """
        issues = []
        imports = []
        used_names = set()
        
        # Collect all imports and used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Check for unused imports
        for imp in imports:
            if imp["type"] == "import":
                name = imp["alias"] or imp["name"].split('.')[0]
                if name not in used_names:
                    issues.append({
                        "type": "unused_import",
                        "import": imp,
                        "message": f"Unused import: {imp['name']}"
                    })
            elif imp["type"] == "from_import":
                name = imp["alias"] or imp["name"]
                if name not in used_names:
                    issues.append({
                        "type": "unused_import",
                        "import": imp,
                        "message": f"Unused import: {imp['name']} from {imp['module']}"
                    })
        
        # Check for duplicate imports
        import_map = {}
        for imp in imports:
            key = (imp["type"], imp.get("module", ""), imp["name"])
            if key in import_map:
                issues.append({
                    "type": "duplicate_import",
                    "import": imp,
                    "message": f"Duplicate import: {imp['name']}"
                })
            else:
                import_map[key] = imp
        
        return issues
    
    def _fix_imports(self, content: str, issues: List[Dict[str, Any]]) -> str:
        """
        Fix import issues in the content.
        
        Args:
            content: Original file content
            issues: List of import issues to fix
            
        Returns:
            str: Fixed content
        """
        lines = content.split('\n')
        
        # Sort issues by line number (descending) to avoid line number shifts
        issues.sort(key=lambda x: x["import"]["line"], reverse=True)
        
        for issue in issues:
            line_num = issue["import"]["line"] - 1  # Convert to 0-based index
            
            if issue["type"] == "unused_import":
                # Remove the unused import line
                if line_num < len(lines):
                    lines.pop(line_num)
            
            elif issue["type"] == "duplicate_import":
                # Remove the duplicate import line
                if line_num < len(lines):
                    lines.pop(line_num)
        
        return '\n'.join(lines)
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Import Fixer Plugin: Processing {len(context.target_files)} files")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Import Fixer Plugin: Fixed {result.issues_fixed} import issues in {result.files_processed} files")
        else:
            print(f"Import Fixer Plugin: Failed to process some files")