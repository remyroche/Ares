#!/usr/bin/env python3
"""
Comprehensive script to fix remaining import conflicts.
This focuses on the high-frequency conflicts that weren't fully resolved.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import shutil
from datetime import datetime
from collections import defaultdict


class ComprehensiveImportFixer:
    """Comprehensive fixer for all import conflicts."""
    
    def __init__(self, report_path: str):
        with open(report_path, 'r') as f:
            self.import_data = json.load(f)
        
        self.backup_dirs = {'syntax_fix_backups', 'syntax_fix_backups_v2'}
        self.changes_made = defaultdict(list)
        self.files_fixed = set()
        
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)
        
        # Skip backup directories
        for part in path.parts:
            if part in self.backup_dirs:
                return False
        
        # Skip if already fixed
        if file_path in self.files_fixed:
            return False
        
        return path.suffix == '.py' and path.exists()
    
    def get_conflicts_for_file(self, file_path: str) -> Dict[str, List[Dict]]:
        """Get all conflicts for a file, grouped by imported name."""
        conflicts_by_name = defaultdict(list)
        
        for issue in self.import_data['results']['issues']['conflicting_imports']:
            if issue['file'] == file_path and self.should_process_file(file_path):
                name = issue['details']['name']
                conflicts_by_name[name].append(issue)
        
        return conflicts_by_name
    
    def fix_system_logger_conflicts(self, file_path: str, content: str, conflicts: List[Dict]) -> str:
        """Fix system_logger import conflicts comprehensively."""
        # Get all modules that system_logger is imported from
        modules = set()
        for conflict in conflicts:
            modules.update(conflict['details']['conflicting_modules'])
        
        # If there's only one actual import, standardize it
        import_lines = []
        for line_num, line in enumerate(content.split('\n'), 1):
            if 'import' in line and 'system_logger' in line:
                import_lines.append((line_num, line))
        
        if len(import_lines) == 1:
            # Standardize to src.utils.logger
            patterns = [
                (r'from\s+logger\s+import\s+system_logger', 'from src.utils.logger import system_logger'),
                (r'from\s+utils\.logger\s+import\s+system_logger', 'from src.utils.logger import system_logger'),
                (r'from\s+\.utils\.logger\s+import\s+system_logger', 'from src.utils.logger import system_logger'),
                (r'from\s+\.\.utils\.logger\s+import\s+system_logger', 'from src.utils.logger import system_logger'),
            ]
            
            for pattern, replacement in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    self.changes_made[file_path].append("Standardized system_logger import to src.utils.logger")
                    break
        else:
            # Multiple imports - need to use aliases
            self.changes_made[file_path].append(f"Multiple system_logger imports detected - manual review needed")
        
        return content
    
    def fix_run_step_conflicts(self, file_path: str, content: str, conflicts: List[Dict]) -> str:
        """Fix run_step import conflicts by adding specific aliases."""
        # Find all run_step imports
        import_pattern = r'from\s+(src\.training\.steps\.[\w_]+)\s+import\s+run_step'
        imports = list(re.finditer(import_pattern, content))
        
        if len(imports) <= 1:
            return content  # No conflict if only one import
        
        # Create aliases based on step number
        step_aliases = {}
        for match in imports:
            module = match.group(1)
            # Extract step number from module name
            step_match = re.search(r'step(\d+)', module)
            if step_match:
                step_num = step_match.group(1)
                alias = f'run_step{step_num}'
                step_aliases[module] = alias
        
        # Apply aliases
        for module, alias in step_aliases.items():
            old_import = f'from {module} import run_step'
            new_import = f'from {module} import run_step as {alias}'
            content = content.replace(old_import, new_import)
            self.changes_made[file_path].append(f"Added alias {alias} for run_step from {module}")
        
        # Update function calls
        # This is a simplified approach - ideally would use AST
        for module, alias in step_aliases.items():
            # Try to identify which run_step calls belong to which module based on context
            # For now, we'll need manual review for this
            self.changes_made[file_path].append(f"Manual review needed: Update run_step() calls to {alias}()")
        
        return content
    
    def fix_generic_conflicts(self, file_path: str, content: str, name: str, conflicts: List[Dict]) -> str:
        """Fix generic import conflicts by adding numbered aliases."""
        modules = []
        for conflict in conflicts:
            modules.extend(conflict['details']['conflicting_modules'])
        modules = list(dict.fromkeys(modules))  # Remove duplicates while preserving order
        
        if len(modules) <= 1:
            return content
        
        # Keep the first import as-is, add aliases for others
        for i, module in enumerate(modules[1:], 1):
            pattern = rf'from\s+{re.escape(module)}\s+import\s+{re.escape(name)}'
            if re.search(pattern, content):
                alias = f'{name}_{module.replace(".", "_").replace("-", "_")}'
                if len(alias) > 50:  # If alias is too long, use numbered version
                    alias = f'{name}_{i+1}'
                
                replacement = f'from {module} import {name} as {alias}'
                content = re.sub(pattern, replacement, content)
                self.changes_made[file_path].append(f"Added alias {alias} for {name} from {module}")
        
        return content
    
    def fix_file(self, file_path: str) -> bool:
        """Fix all import conflicts in a file."""
        if not self.should_process_file(file_path):
            return False
        
        conflicts_by_name = self.get_conflicts_for_file(file_path)
        if not conflicts_by_name:
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply specific fixes for known conflict patterns
            for name, conflicts in conflicts_by_name.items():
                if name == 'system_logger':
                    content = self.fix_system_logger_conflicts(file_path, content, conflicts)
                elif name == 'run_step':
                    content = self.fix_run_step_conflicts(file_path, content, conflicts)
                else:
                    content = self.fix_generic_conflicts(file_path, content, name, conflicts)
            
            # Write back if changed
            if content != original_content:
                # Create backup
                backup_dir = Path('comprehensive_import_fix_backups')
                backup_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{Path(file_path).stem}_{timestamp}.backup"
                backup_path = backup_dir / backup_name
                
                shutil.copy2(file_path, backup_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_fixed.add(file_path)
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False


def main():
    """Main function to apply comprehensive import fixes."""
    print("Comprehensive Import Conflict Fixer")
    print("=" * 50)
    
    # Use the import analysis report
    report_path = 'sequential_fixer_reports/import_analysis_report_20250903_115608.json'
    if not os.path.exists(report_path):
        print(f"Error: Report not found at {report_path}")
        return
    
    fixer = ComprehensiveImportFixer(report_path)
    
    # Get all unique files with conflicts
    files_with_conflicts = set()
    for issue in fixer.import_data['results']['issues']['conflicting_imports']:
        if fixer.should_process_file(issue['file']):
            files_with_conflicts.add(issue['file'])
    
    print(f"Found {len(files_with_conflicts)} files with import conflicts")
    
    # Process files
    fixed_count = 0
    for i, file_path in enumerate(sorted(files_with_conflicts)):
        if i % 50 == 0 and i > 0:
            print(f"Progress: {i}/{len(files_with_conflicts)} files...")
        
        if fixer.fix_file(file_path):
            fixed_count += 1
    
    # Generate detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'files_processed': len(files_with_conflicts),
        'files_fixed': fixed_count,
        'changes_by_file': dict(fixer.changes_made),
        'files_requiring_manual_review': []
    }
    
    # Identify files needing manual review
    for file_path, changes in fixer.changes_made.items():
        if any('manual review needed' in change.lower() for change in changes):
            report['files_requiring_manual_review'].append(file_path)
    
    # Save report
    report_path = f'comprehensive_import_fix_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Fix Summary:")
    print(f"  Files processed: {len(files_with_conflicts)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files requiring manual review: {len(report['files_requiring_manual_review'])}")
    print(f"  Report saved to: {report_path}")
    
    # Show files needing manual review
    if report['files_requiring_manual_review']:
        print("\nFiles requiring manual review:")
        for file_path in report['files_requiring_manual_review'][:10]:
            print(f"  - {file_path}")
        if len(report['files_requiring_manual_review']) > 10:
            print(f"  ... and {len(report['files_requiring_manual_review']) - 10} more")
    
    print("\nBackups created in: comprehensive_import_fix_backups/")
    
    # Show change statistics
    change_types = defaultdict(int)
    for changes in fixer.changes_made.values():
        for change in changes:
            if 'Standardized' in change:
                change_types['standardized'] += 1
            elif 'Added alias' in change:
                change_types['aliased'] += 1
            elif 'manual review' in change.lower():
                change_types['manual_review'] += 1
    
    print("\nChange statistics:")
    for change_type, count in change_types.items():
        print(f"  {change_type}: {count}")


if __name__ == "__main__":
    main()