#!/usr/bin/env python3
"""
Targeted script to fix the most common import conflicts and signature issues.
Focuses on the top issues identified in the analysis.
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import shutil
from datetime import datetime
from collections import defaultdict


class TargetedImportFixer:
    """Fixes the most common import conflicts with targeted solutions."""
    
    def __init__(self):
        self.backup_dirs = {'syntax_fix_backups', 'syntax_fix_backups_v2'}
        self.changes_made = defaultdict(list)
        
        # Define specific resolution strategies for common conflicts
        self.resolution_strategies = {
            'system_logger': self._fix_system_logger_imports,
            'run_step': self._fix_run_step_imports,
            'handles_errors': self._fix_handles_errors_imports,
            'get_default_config': self._fix_get_default_config_imports,
            'Callable': self._fix_callable_imports,
        }
    
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)
        
        # Skip backup directories
        for part in path.parts:
            if part in self.backup_dirs:
                return False
        
        # Only process Python files
        return path.suffix == '.py' and path.exists()
    
    def _fix_system_logger_imports(self, file_path: str, content: str) -> str:
        """Fix system_logger import conflicts."""
        # Standardize all system_logger imports to use src.utils.logger
        patterns = [
            (r'from\s+logger\s+import\s+system_logger', 'from src.utils.logger import system_logger'),
            (r'from\s+utils\.logger\s+import\s+system_logger', 'from src.utils.logger import system_logger'),
            (r'import\s+logger\.system_logger', 'from src.utils.logger import system_logger'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.changes_made[file_path].append(f"Standardized system_logger import to src.utils.logger")
        
        return content
    
    def _fix_run_step_imports(self, file_path: str, content: str) -> str:
        """Fix run_step import conflicts by adding aliases."""
        # Map of module to alias
        aliases = {
            'src.training.steps.step2_feature_engineering': 'run_step2',
            'src.training.steps.step3_hmm_regime_discovery': 'run_step3',
            'src.training.steps.step6_feature_engineering': 'run_step6',
            'src.training.steps.step7_enhanced_matrix_operations': 'run_step7',
        }
        
        # Find all run_step imports in the file
        import_pattern = r'from\s+(src\.training\.steps\.\w+)\s+import\s+run_step'
        imports_found = re.findall(import_pattern, content)
        
        if len(imports_found) > 1:
            # Multiple run_step imports - apply aliases
            for module in imports_found:
                if module in aliases:
                    alias = aliases[module]
                    old_import = f'from {module} import run_step'
                    new_import = f'from {module} import run_step as {alias}'
                    content = content.replace(old_import, new_import)
                    
                    # Update usage in the code
                    # Look for run_step calls and replace with alias
                    content = self._update_function_calls(content, 'run_step', alias, context_module=module)
                    
                    self.changes_made[file_path].append(f"Added alias {alias} for run_step from {module}")
        
        return content
    
    def _fix_handles_errors_imports(self, file_path: str, content: str) -> str:
        """Fix handles_errors decorator import conflicts."""
        aliases = {
            'src.core.decorators': 'core_handles_errors',
            'src.utils.decorators': 'utils_handles_errors',
            'decorators': 'handles_errors',
        }
        
        import_pattern = r'from\s+([\w\.]+)\s+import\s+handles_errors'
        imports_found = re.findall(import_pattern, content)
        
        if len(imports_found) > 1:
            for module in imports_found:
                if module in aliases:
                    alias = aliases[module]
                    old_import = f'from {module} import handles_errors'
                    new_import = f'from {module} import handles_errors as {alias}'
                    content = content.replace(old_import, new_import)
                    
                    # Update decorator usage
                    content = self._update_decorator_usage(content, 'handles_errors', alias)
                    
                    self.changes_made[file_path].append(f"Added alias {alias} for handles_errors from {module}")
        
        return content
    
    def _fix_get_default_config_imports(self, file_path: str, content: str) -> str:
        """Fix get_default_config import conflicts."""
        # Standardize to use src.config
        patterns = [
            (r'from\s+config\s+import\s+get_default_config', 'from src.config import get_default_config'),
            (r'from\s+\.config\s+import\s+get_default_config', 'from src.config import get_default_config'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.changes_made[file_path].append("Standardized get_default_config import to src.config")
        
        return content
    
    def _fix_callable_imports(self, file_path: str, content: str) -> str:
        """Fix Callable type import conflicts."""
        # Standardize to use typing.Callable
        if 'from typing import' in content and 'Callable' in content:
            # Already has typing import, ensure Callable is included
            typing_import_pattern = r'from\s+typing\s+import\s+([^\\n]+)'
            match = re.search(typing_import_pattern, content)
            if match and 'Callable' not in match.group(1):
                imports = match.group(1)
                new_imports = f"{imports}, Callable"
                content = content.replace(f"from typing import {imports}", f"from typing import {new_imports}")
                self.changes_made[file_path].append("Added Callable to typing imports")
        
        # Remove any collections.abc.Callable imports if typing.Callable exists
        if 'from typing import' in content and 'Callable' in content:
            content = re.sub(r'from\s+collections\.abc\s+import\s+Callable\s*\n', '', content)
            self.changes_made[file_path].append("Removed collections.abc.Callable in favor of typing.Callable")
        
        return content
    
    def _update_function_calls(self, content: str, old_name: str, new_name: str, context_module: str = '') -> str:
        """Update function calls to use the new aliased name."""
        # This is a simplified approach - a full AST-based solution would be more robust
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            # Skip import lines
            if 'import' in line and old_name in line:
                new_lines.append(line)
            else:
                # Replace function calls
                line = re.sub(rf'\b{old_name}\s*\(', f'{new_name}(', line)
                new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def _update_decorator_usage(self, content: str, old_name: str, new_name: str) -> str:
        """Update decorator usage to use the new aliased name."""
        # Replace @decorator usage
        content = re.sub(rf'^(\s*)@{old_name}\b', rf'\1@{new_name}', content, flags=re.MULTILINE)
        return content
    
    def fix_file(self, file_path: str) -> bool:
        """Fix import conflicts in a single file."""
        if not self.should_process_file(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply all resolution strategies
            for conflict_name, fix_function in self.resolution_strategies.items():
                if conflict_name in content:
                    content = fix_function(file_path, content)
            
            # Write back if changed
            if content != original_content:
                # Create backup
                backup_dir = Path('import_fix_backups')
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"{Path(file_path).name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup"
                shutil.copy2(file_path, backup_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False


class TargetedSignatureFixer:
    """Fixes the most common function signature issues."""
    
    def __init__(self):
        self.backup_dirs = {'syntax_fix_backups', 'syntax_fix_backups_v2'}
        self.changes_made = defaultdict(list)
    
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)
        
        # Skip backup directories
        for part in path.parts:
            if part in self.backup_dirs:
                return False
        
        return path.suffix == '.py' and path.exists()
    
    def fix_missing_self_in_method_calls(self, file_path: str, content: str) -> str:
        """Fix method calls that are missing 'self.'"""
        try:
            tree = ast.parse(content)
            
            class SelfMethodCallFixer(ast.NodeTransformer):
                def __init__(self):
                    self.in_class = False
                    self.class_methods = set()
                    self.changes = []
                
                def visit_ClassDef(self, node):
                    old_in_class = self.in_class
                    self.in_class = True
                    
                    # Collect method names
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            self.class_methods.add(item.name)
                    
                    self.generic_visit(node)
                    self.in_class = old_in_class
                    return node
                
                def visit_Call(self, node):
                    if (self.in_class and 
                        isinstance(node.func, ast.Name) and 
                        node.func.id in self.class_methods):
                        # This is a call to a class method without self
                        self.changes.append((node.lineno, node.func.id))
                    
                    self.generic_visit(node)
                    return node
            
            fixer = SelfMethodCallFixer()
            fixer.visit(tree)
            
            if fixer.changes:
                lines = content.split('\n')
                for line_no, method_name in sorted(fixer.changes, reverse=True):
                    if 0 <= line_no - 1 < len(lines):
                        line = lines[line_no - 1]
                        # Add self. prefix if not already present
                        if f'self.{method_name}' not in line:
                            lines[line_no - 1] = re.sub(rf'\b{method_name}\s*\(', f'self.{method_name}(', line)
                            self.changes_made[file_path].append(f"Added self. to {method_name} call at line {line_no}")
                
                content = '\n'.join(lines)
        
        except SyntaxError:
            # If we can't parse the file, skip AST-based fixes
            pass
        
        return content
    
    def fix_file(self, file_path: str) -> bool:
        """Fix signature issues in a single file."""
        if not self.should_process_file(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply fixes
            content = self.fix_missing_self_in_method_calls(file_path, content)
            
            # Write back if changed
            if content != original_content:
                # Create backup
                backup_dir = Path('signature_fix_backups')
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"{Path(file_path).name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup"
                shutil.copy2(file_path, backup_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False


def find_files_with_issues():
    """Find files with import and signature issues from the reports."""
    files_to_fix = set()
    
    # Load import report
    import_report_path = 'sequential_fixer_reports/import_analysis_report_20250903_115608.json'
    if os.path.exists(import_report_path):
        with open(import_report_path, 'r') as f:
            import_data = json.load(f)
        
        for issue in import_data['results']['issues']['conflicting_imports']:
            file_path = issue['file']
            if not any(backup in file_path for backup in ['syntax_fix_backups', 'syntax_fix_backups_v2']):
                files_to_fix.add(file_path)
    
    # Load signature report
    signature_report_path = 'sequential_fixer_reports/signature_analysis_report_20250903_115608.json'
    if os.path.exists(signature_report_path):
        with open(signature_report_path, 'r') as f:
            signature_data = json.load(f)
        
        for issue in signature_data['results']['issues']['compatibility_issues']:
            file_path = issue['file']
            if not any(backup in file_path for backup in ['syntax_fix_backups', 'syntax_fix_backups_v2']):
                files_to_fix.add(file_path)
    
    return sorted(files_to_fix)


def main():
    """Main function to apply targeted fixes."""
    print("Targeted Import and Signature Fixer")
    print("=" * 50)
    
    # Find files with issues
    files_to_fix = find_files_with_issues()
    print(f"Found {len(files_to_fix)} files with issues (excluding backup directories)")
    
    # Create fixers
    import_fixer = TargetedImportFixer()
    signature_fixer = TargetedSignatureFixer()
    
    # Track progress
    import_fixes = 0
    signature_fixes = 0
    total_processed = 0
    
    # Process files
    for i, file_path in enumerate(files_to_fix):
        if i % 50 == 0:
            print(f"\nProgress: {i}/{len(files_to_fix)} files...")
        
        total_processed += 1
        
        # Fix imports
        if import_fixer.fix_file(file_path):
            import_fixes += 1
        
        # Fix signatures
        if signature_fixer.fix_file(file_path):
            signature_fixes += 1
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'files_processed': total_processed,
        'import_fixes': import_fixes,
        'signature_fixes': signature_fixes,
        'import_changes': dict(import_fixer.changes_made),
        'signature_changes': dict(signature_fixer.changes_made)
    }
    
    report_path = f'targeted_fix_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Fix Summary:")
    print(f"  Files processed: {total_processed}")
    print(f"  Files with import fixes: {import_fixes}")
    print(f"  Files with signature fixes: {signature_fixes}")
    print(f"  Report saved to: {report_path}")
    
    # Show top changes
    if import_fixer.changes_made:
        print("\nTop files with import changes:")
        sorted_changes = sorted(import_fixer.changes_made.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:5]
        for file_path, changes in sorted_changes:
            print(f"  {file_path}: {len(changes)} changes")
            for change in changes[:2]:
                print(f"    - {change}")
    
    if signature_fixer.changes_made:
        print("\nTop files with signature changes:")
        sorted_changes = sorted(signature_fixer.changes_made.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:5]
        for file_path, changes in sorted_changes:
            print(f"  {file_path}: {len(changes)} changes")
            for change in changes[:2]:
                print(f"    - {change}")
    
    print("\nBackups created in:")
    print("  - import_fix_backups/")
    print("  - signature_fix_backups/")


if __name__ == "__main__":
    main()