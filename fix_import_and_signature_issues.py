#!/usr/bin/env python3
"""
Script to fix import conflicts and function signature compatibility issues.
This script will:
1. Fix import conflicts by using aliases or qualified imports
2. Fix function signature compatibility issues
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import shutil
from datetime import datetime


class ImportConflictFixer:
    """Fixes import conflicts by adding aliases or converting to qualified imports."""
    
    def __init__(self, import_report_path: str):
        """Initialize with import analysis report."""
        with open(import_report_path, 'r') as f:
            self.import_data = json.load(f)
        
        # Common import conflict resolution mappings
        self.import_aliases = {
            'system_logger': {
                'src.utils.logger': 'src_system_logger',
                'logger': 'logger_system_logger',
                'utils.logger': 'utils_system_logger'
            },
            'run_step': {
                'src.training.steps.step2_feature_engineering': 'run_step_feature_eng',
                'src.training.steps.step7_enhanced_matrix_operations': 'run_step_matrix_ops',
                'src.training.steps.step3_hmm_regime_discovery': 'run_step_hmm',
                'src.training.steps.step6_feature_engineering': 'run_step_feature_eng6'
            },
            'handles_errors': {
                'src.core.decorators': 'core_handles_errors',
                'src.utils.decorators': 'utils_handles_errors',
                'decorators': 'handles_errors_decorator'
            }
        }
        
        self.backup_dirs = ['./syntax_fix_backups/', './syntax_fix_backups_v2/']
        
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        # Skip backup directories
        for backup_dir in self.backup_dirs:
            if file_path.startswith(backup_dir):
                return False
        
        # Only process Python files
        if not file_path.endswith('.py'):
            return False
            
        return True
    
    def get_conflicts_for_file(self, file_path: str) -> List[Dict]:
        """Get all import conflicts for a specific file."""
        conflicts = []
        for issue in self.import_data['results']['issues']['conflicting_imports']:
            if issue['file'] == file_path and self.should_process_file(file_path):
                conflicts.append(issue)
        return conflicts
    
    def fix_import_in_file(self, file_path: str, conflicts: List[Dict]) -> bool:
        """Fix import conflicts in a single file."""
        if not conflicts:
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Group conflicts by the imported name
            conflicts_by_name = {}
            for conflict in conflicts:
                name = conflict['details']['name']
                if name not in conflicts_by_name:
                    conflicts_by_name[name] = []
                conflicts_by_name[name].append(conflict)
            
            # Process each conflicting name
            for name, name_conflicts in conflicts_by_name.items():
                if name not in self.import_aliases:
                    # Create generic aliases
                    modules = name_conflicts[0]['details']['conflicting_modules']
                    for i, module in enumerate(modules[:3]):  # Handle up to 3 conflicts
                        if i == 0:
                            # Keep first import as-is
                            continue
                        else:
                            # Create alias for subsequent imports
                            alias = f"{name}_{i+1}"
                            # Update import statement
                            pattern = rf"from\s+{re.escape(module)}\s+import\s+{re.escape(name)}"
                            replacement = f"from {module} import {name} as {alias}"
                            content = re.sub(pattern, replacement, content)
                            
                            # Update usage in code
                            # This is simplified - a more sophisticated approach would use AST
                            content = self._update_usage_with_alias(content, name, alias, module)
                else:
                    # Use predefined aliases
                    alias_map = self.import_aliases[name]
                    for module, alias in alias_map.items():
                        pattern = rf"from\s+{re.escape(module)}\s+import\s+{re.escape(name)}"
                        replacement = f"from {module} import {name} as {alias}"
                        content = re.sub(pattern, replacement, content)
                        
                        # Update usage
                        content = self._update_usage_with_alias(content, name, alias, module)
            
            # Write back if changed
            if content != original_content:
                # Create backup
                backup_path = file_path + '.import_fix_backup'
                shutil.copy2(file_path, backup_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False
    
    def _update_usage_with_alias(self, content: str, old_name: str, new_alias: str, module: str) -> str:
        """Update usage of imported name with alias."""
        # This is a simplified implementation
        # A full implementation would use AST to properly identify usage contexts
        
        # Don't replace in import statements
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            if 'import' in line and old_name in line:
                new_lines.append(line)
            else:
                # Simple replacement - may need refinement for complex cases
                new_line = line.replace(f"{old_name}(", f"{new_alias}(")
                new_line = new_line.replace(f"{old_name}.", f"{new_alias}.")
                new_line = new_line.replace(f" {old_name} ", f" {new_alias} ")
                new_lines.append(new_line)
        
        return '\n'.join(new_lines)


class SignatureCompatibilityFixer:
    """Fixes function signature compatibility issues."""
    
    def __init__(self, signature_report_path: str):
        """Initialize with signature analysis report."""
        with open(signature_report_path, 'r') as f:
            self.signature_data = json.load(f)
        
        self.backup_dirs = ['./syntax_fix_backups/', './syntax_fix_backups_v2/']
    
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        for backup_dir in self.backup_dirs:
            if file_path.startswith(backup_dir):
                return False
        return file_path.endswith('.py')
    
    def get_issues_for_file(self, file_path: str) -> List[Dict]:
        """Get all signature issues for a specific file."""
        issues = []
        for issue in self.signature_data['results']['issues']['compatibility_issues']:
            if issue['file'] == file_path and self.should_process_file(file_path):
                issues.append(issue)
        return issues
    
    def fix_signatures_in_file(self, file_path: str, issues: List[Dict]) -> bool:
        """Fix signature compatibility issues in a file."""
        if not issues:
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            lines = content.split('\n')
            
            # Sort issues by line number in reverse order to avoid offset issues
            sorted_issues = sorted(issues, key=lambda x: x['line'], reverse=True)
            
            for issue in sorted_issues:
                line_num = issue['line'] - 1  # Convert to 0-based
                if 0 <= line_num < len(lines):
                    line = lines[line_num]
                    
                    if issue['message'].startswith('Missing required arguments:'):
                        # Fix missing self argument in method calls
                        func_name = issue['details']['function_name']
                        
                        # Check if it's a method call that's missing 'self'
                        if 'self' in issue['details']['definition']['args'] and 'self' not in issue['details']['call']['args']:
                            # This is likely a method being called without self
                            # Convert to self.method() format
                            pattern = rf"\b{func_name}\s*\("
                            replacement = f"self.{func_name}("
                            lines[line_num] = re.sub(pattern, replacement, line)
            
            # Reconstruct content
            new_content = '\n'.join(lines)
            
            if new_content != original_content:
                # Create backup
                backup_path = file_path + '.signature_fix_backup'
                shutil.copy2(file_path, backup_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False


def main():
    """Main function to fix import and signature issues."""
    
    # Paths to the analysis reports
    import_report = 'sequential_fixer_reports/import_analysis_report_20250903_115608.json'
    signature_report = 'sequential_fixer_reports/signature_analysis_report_20250903_115608.json'
    
    # Create fixers
    import_fixer = ImportConflictFixer(import_report)
    signature_fixer = SignatureCompatibilityFixer(signature_report)
    
    # Get all files with issues
    files_with_import_issues = set()
    for issue in import_fixer.import_data['results']['issues']['conflicting_imports']:
        if import_fixer.should_process_file(issue['file']):
            files_with_import_issues.add(issue['file'])
    
    files_with_signature_issues = set()
    for issue in signature_fixer.signature_data['results']['issues']['compatibility_issues']:
        if signature_fixer.should_process_file(issue['file']):
            files_with_signature_issues.add(issue['file'])
    
    print(f"Files with import conflicts: {len(files_with_import_issues)}")
    print(f"Files with signature issues: {len(files_with_signature_issues)}")
    
    # Create a report
    report = {
        'timestamp': datetime.now().isoformat(),
        'import_fixes': [],
        'signature_fixes': [],
        'errors': []
    }
    
    # Fix import conflicts
    print("\nFixing import conflicts...")
    for i, file_path in enumerate(sorted(files_with_import_issues)):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(files_with_import_issues)}")
        
        conflicts = import_fixer.get_conflicts_for_file(file_path)
        if import_fixer.fix_import_in_file(file_path, conflicts):
            report['import_fixes'].append({
                'file': file_path,
                'conflicts_fixed': len(conflicts)
            })
    
    # Fix signature issues
    print("\nFixing signature compatibility issues...")
    for i, file_path in enumerate(sorted(files_with_signature_issues)):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(files_with_signature_issues)}")
        
        issues = signature_fixer.get_issues_for_file(file_path)
        if signature_fixer.fix_signatures_in_file(file_path, issues):
            report['signature_fixes'].append({
                'file': file_path,
                'issues_fixed': len(issues)
            })
    
    # Save report
    report_path = f'fix_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFix complete!")
    print(f"Import fixes applied: {len(report['import_fixes'])}")
    print(f"Signature fixes applied: {len(report['signature_fixes'])}")
    print(f"Report saved to: {report_path}")
    
    # Show summary of most common fixes
    if report['import_fixes']:
        print("\nTop files with import fixes:")
        sorted_import_fixes = sorted(report['import_fixes'], 
                                   key=lambda x: x['conflicts_fixed'], 
                                   reverse=True)
        for fix in sorted_import_fixes[:5]:
            print(f"  {fix['file']}: {fix['conflicts_fixed']} conflicts fixed")
    
    if report['signature_fixes']:
        print("\nTop files with signature fixes:")
        sorted_sig_fixes = sorted(report['signature_fixes'], 
                                key=lambda x: x['issues_fixed'], 
                                reverse=True)
        for fix in sorted_sig_fixes[:5]:
            print(f"  {fix['file']}: {fix['issues_fixed']} issues fixed")


if __name__ == "__main__":
    main()