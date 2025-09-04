"""
Syntax Fixer Plugin Example

Demonstrates how to create a plugin for syntax fixing functionality.
"""

import ast
import tempfile
from pathlib import Path
from typing import Dict, Any, Set
from code_quality.plugins.base_plugin import FileProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class SyntaxFixerPlugin(FileProcessorPlugin):
    """
    Plugin for fixing Python syntax errors.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="syntax_fixer",
            version="1.0.0",
            description="Fixes common Python syntax errors",
            author="Code Quality Team",
            category=PluginCategory.SYNTAX,
            priority=PluginPriority.HIGH,
            dependencies=[],
            tags={"syntax", "fixing", "python"},
            required_packages=[],
            optional_packages=["autopep8", "black"],
            configuration_schema={
                "fix_indentation": {"type": "boolean", "default": True},
                "fix_imports": {"type": "boolean", "default": True},
                "fix_quotes": {"type": "boolean", "default": False},
                "max_line_length": {"type": "integer", "default": 120}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        # This plugin only requires standard library
        return True
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi'}
    
    def process_file(self, file_path: Path, context) -> Dict[str, Any]:
        """
        Process a single Python file for syntax errors.
        
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
            
            # Check for syntax errors
            try:
                ast.parse(content)
                # No syntax errors found
                return result
            except SyntaxError as e:
                result["issues_found"] = 1
                
                # Try to fix common syntax errors
                if not context.dry_run:
                    fixed_content = self._fix_syntax_errors(content, e)
                    if fixed_content != content:
                        # Write fixed content back
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        result["issues_fixed"] = 1
                        result["success"] = True
                    else:
                        result["success"] = False
                        result["error"] = f"Could not fix syntax error: {e}"
                else:
                    result["success"] = False
                    result["error"] = f"Syntax error found (dry run): {e}"
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _fix_syntax_errors(self, content: str, syntax_error: SyntaxError) -> str:
        """
        Attempt to fix common syntax errors.
        
        Args:
            content: Original file content
            syntax_error: The syntax error that occurred
            
        Returns:
            str: Fixed content (or original if couldn't fix)
        """
        lines = content.split('\n')
        error_line = syntax_error.lineno - 1  # Convert to 0-based index
        
        if error_line >= len(lines):
            return content
        
        error_text = syntax_error.text
        error_msg = str(syntax_error.msg)
        
        # Fix common issues
        if "unexpected EOF while parsing" in error_msg:
            # Try to add missing closing parenthesis/bracket
            line = lines[error_line]
            if line.count('(') > line.count(')'):
                lines[error_line] = line + ')'
            elif line.count('[') > line.count(']'):
                lines[error_line] = line + ']'
            elif line.count('{') > line.count('}'):
                lines[error_line] = line + '}'
        
        elif "invalid syntax" in error_msg:
            # Try to fix common invalid syntax issues
            line = lines[error_line]
            
            # Fix missing colon after if/for/while/def/class
            if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'def ', 'class ']):
                if not line.strip().endswith(':'):
                    lines[error_line] = line.rstrip() + ':'
            
            # Fix assignment in condition
            if ' = ' in line and ('if ' in line or 'while ' in line):
                lines[error_line] = line.replace(' = ', ' == ')
        
        elif "expected an indented block" in error_msg:
            # Add pass statement for empty blocks
            if error_line + 1 < len(lines):
                next_line = lines[error_line + 1]
                if not next_line.strip():
                    lines.insert(error_line + 1, '    pass')
            else:
                lines.append('    pass')
        
        return '\n'.join(lines)
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Syntax Fixer Plugin: Processing {len(context.target_files)} files")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Syntax Fixer Plugin: Fixed {result.issues_fixed} syntax errors in {result.files_processed} files")
        else:
            print(f"Syntax Fixer Plugin: Failed to process some files")