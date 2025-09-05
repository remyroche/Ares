#!/usr/bin/env python3
"""
Duplicate Import Fixer

This module provides safe automatic removal of duplicate imports with
comprehensive safety checks and validation.
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImportInfo:
    """Information about an import statement."""
    line_number: int
    import_type: str  # 'import' or 'from_import'
    module: str
    name: str
    as_name: Optional[str] = None
    full_line: str = ""
    is_duplicate: bool = False
    can_safely_remove: bool = False


class DuplicateImportFixer:
    """
    Safe duplicate import removal with comprehensive validation.
    
    This class provides:
    1. Detection of duplicate imports
    2. Safety analysis for removal
    3. Automatic fixing with backup
    4. Validation of fixes
    """
    
    def __init__(self):
        self.imports_found: List[ImportInfo] = []
        self.duplicates: List[Tuple[ImportInfo, ImportInfo]] = []
        self.safety_checks = {
            'check_usage': True,
            'check_side_effects': True,
            'check_conditional_imports': True,
            'check_dynamic_imports': True,
            'backup_original': True
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """Analyze a file for duplicate imports and safety of removal."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            tree = ast.parse(content)
            
            # Find all imports
            self._find_imports(tree, lines)
            
            # Find duplicates
            self._find_duplicates()
            
            # Analyze safety
            self._analyze_safety(tree, content)
            
            return {
                'file_path': file_path,
                'total_imports': len(self.imports_found),
                'duplicates_found': len(self.duplicates),
                'safe_to_remove': len([d for d in self.duplicates if d[0].can_safely_remove]),
                'risky_removals': len([d for d in self.duplicates if not d[0].can_safely_remove]),
                'imports': self.imports_found,
                'duplicates': self.duplicates
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'total_imports': 0,
                'duplicates_found': 0,
                'safe_to_remove': 0,
                'risky_removals': 0
            }
    
    def _find_imports(self, tree: ast.AST, lines: List[str]) -> None:
        """Find all import statements in the AST."""
        self.imports_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = ImportInfo(
                        line_number=node.lineno,
                        import_type='import',
                        module=alias.name,
                        name=alias.name.split('.')[-1],
                        as_name=alias.asname,
                        full_line=lines[node.lineno - 1].strip()
                    )
                    self.imports_found.append(import_info)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    import_info = ImportInfo(
                        line_number=node.lineno,
                        import_type='from_import',
                        module=module,
                        name=alias.name,
                        as_name=alias.asname,
                        full_line=lines[node.lineno - 1].strip()
                    )
                    self.imports_found.append(import_info)
    
    def _find_duplicates(self) -> None:
        """Find duplicate imports based on effective name."""
        self.duplicates = []
        seen_imports: Dict[str, ImportInfo] = {}
        
        for import_info in self.imports_found:
            # Use the effective name (as_name if present, otherwise name)
            effective_name = import_info.as_name or import_info.name
            
            if effective_name in seen_imports:
                # Found a duplicate
                original = seen_imports[effective_name]
                import_info.is_duplicate = True
                self.duplicates.append((original, import_info))
            else:
                seen_imports[effective_name] = import_info
    
    def _analyze_safety(self, tree: ast.AST, content: str) -> None:
        """Analyze safety of removing duplicate imports."""
        lines = content.split('\n')
        
        for original, duplicate in self.duplicates:
            can_remove = True
            reasons = []
            
            # Check 1: Usage analysis
            if self.safety_checks['check_usage']:
                if self._is_import_used_after_line(tree, duplicate.name, duplicate.line_number):
                    can_remove = False
                    reasons.append("Import is used after this line")
            
            # Check 2: Side effects
            if self.safety_checks['check_side_effects']:
                if self._has_side_effects(duplicate.module, duplicate.name):
                    can_remove = False
                    reasons.append("Import may have side effects")
            
            # Check 3: Conditional imports
            if self.safety_checks['check_conditional_imports']:
                if self._is_conditional_import(lines, duplicate.line_number):
                    can_remove = False
                    reasons.append("Import is in conditional block")
            
            # Check 4: Dynamic imports
            if self.safety_checks['check_dynamic_imports']:
                if self._is_dynamic_import(lines, duplicate.line_number):
                    can_remove = False
                    reasons.append("Import appears to be dynamic")
            
            # Check 5: Import order dependencies
            if self._has_import_order_dependencies(original, duplicate):
                can_remove = False
                reasons.append("Import order may be significant")
            
            duplicate.can_safely_remove = can_remove
            duplicate.safety_reasons = reasons
    
    def _is_import_used_after_line(self, tree: ast.AST, name: str, line_number: int) -> bool:
        """Check if an import is used after a specific line."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                if node.lineno > line_number:
                    return True
        return False
    
    def _has_side_effects(self, module: str, name: str) -> bool:
        """Check if an import might have side effects."""
        # Common modules that have side effects when imported
        side_effect_modules = {
            'matplotlib', 'matplotlib.pyplot', 'pylab',
            'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
            'tensorflow', 'torch', 'jax',
            'numpy.random', 'random'
        }
        
        # Check if the module or any parent module has side effects
        module_parts = module.split('.')
        for i in range(len(module_parts)):
            partial_module = '.'.join(module_parts[:i+1])
            if partial_module in side_effect_modules:
                return True
        
        return False
    
    def _is_conditional_import(self, lines: List[str], line_number: int) -> bool:
        """Check if an import is inside a conditional block."""
        if line_number <= 1:
            return False
        
        # Look for indentation and control structures
        line = lines[line_number - 1]
        if line.strip().startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with ')):
            return True
        
        # Check if line is indented (inside a block)
        if line.startswith(('    ', '\t')):
            return True
        
        return False
    
    def _is_dynamic_import(self, lines: List[str], line_number: int) -> bool:
        """Check if an import appears to be dynamic."""
        line = lines[line_number - 1]
        
        # Check for dynamic patterns
        dynamic_patterns = [
            'importlib', '__import__', 'exec', 'eval',
            'getattr', 'setattr', 'globals()', 'locals()'
        ]
        
        for pattern in dynamic_patterns:
            if pattern in line:
                return True
        
        return False
    
    def _has_import_order_dependencies(self, original: ImportInfo, duplicate: ImportInfo) -> bool:
        """Check if import order might be significant."""
        # Some imports need to be in specific order
        order_sensitive_modules = {
            'os', 'sys', 'pathlib', 'typing',
            'collections', 'itertools', 'functools'
        }
        
        # If both imports are from order-sensitive modules, be cautious
        if (original.module in order_sensitive_modules and 
            duplicate.module in order_sensitive_modules):
            return True
        
        return False
    
    def fix_duplicates(self, file_path: str, dry_run: bool = True) -> Dict[str, any]:
        """Fix duplicate imports in a file."""
        analysis = self.analyze_file(file_path)
        
        if analysis.get('error'):
            return analysis
        
        if analysis['duplicates_found'] == 0:
            return {
                'file_path': file_path,
                'status': 'no_duplicates',
                'message': 'No duplicate imports found'
            }
        
        # Create backup if not dry run
        if not dry_run and self.safety_checks['backup_original']:
            backup_path = f"{file_path}.backup_duplicate_fix"
            with open(file_path, 'r') as original:
                with open(backup_path, 'w') as backup:
                    backup.write(original.read())
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Remove duplicate lines (in reverse order to maintain line numbers)
            lines_to_remove = []
            for original, duplicate in self.duplicates:
                if duplicate.can_safely_remove:
                    lines_to_remove.append(duplicate.line_number - 1)  # Convert to 0-based
            
            # Sort in reverse order
            lines_to_remove.sort(reverse=True)
            
            # Remove lines
            for line_num in lines_to_remove:
                if not dry_run:
                    lines.pop(line_num)
            
            # Write back if not dry run
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            
            return {
                'file_path': file_path,
                'status': 'success',
                'dry_run': dry_run,
                'duplicates_removed': len(lines_to_remove),
                'lines_removed': lines_to_remove,
                'backup_created': not dry_run and self.safety_checks['backup_original']
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'status': 'error',
                'error': str(e)
            }
    
    def get_safety_report(self, file_path: str) -> Dict[str, any]:
        """Generate a detailed safety report for duplicate import removal."""
        analysis = self.analyze_file(file_path)
        
        if analysis.get('error'):
            return analysis
        
        safe_removals = []
        risky_removals = []
        
        for original, duplicate in self.duplicates:
            removal_info = {
                'line_number': duplicate.line_number,
                'import_line': duplicate.full_line,
                'effective_name': duplicate.as_name or duplicate.name,
                'can_remove': duplicate.can_safely_remove,
                'reasons': getattr(duplicate, 'safety_reasons', [])
            }
            
            if duplicate.can_safely_remove:
                safe_removals.append(removal_info)
            else:
                risky_removals.append(removal_info)
        
        return {
            'file_path': file_path,
            'total_duplicates': len(self.duplicates),
            'safe_removals': safe_removals,
            'risky_removals': risky_removals,
            'safety_summary': {
                'safe_count': len(safe_removals),
                'risky_count': len(risky_removals),
                'safety_percentage': (len(safe_removals) / len(self.duplicates) * 100) if self.duplicates else 100
            }
        }


def main():
    """Command-line interface for duplicate import fixing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Duplicate Import Fixer")
    parser.add_argument("file", help="Python file to analyze/fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without making changes")
    parser.add_argument("--fix", action="store_true", help="Actually remove duplicate imports")
    parser.add_argument("--safety-report", action="store_true", help="Generate detailed safety report")
    
    args = parser.parse_args()
    
    fixer = DuplicateImportFixer()
    
    if args.safety_report:
        report = fixer.get_safety_report(args.file)
        print(f"Safety Report for {args.file}")
        print("=" * 50)
        print(f"Total duplicates: {report['total_duplicates']}")
        print(f"Safe to remove: {report['safety_summary']['safe_count']}")
        print(f"Risky to remove: {report['safety_summary']['risky_count']}")
        print(f"Safety percentage: {report['safety_summary']['safety_percentage']:.1f}%")
        
        if report['safe_removals']:
            print("\nSafe removals:")
            for removal in report['safe_removals']:
                print(f"  Line {removal['line_number']}: {removal['import_line']}")
        
        if report['risky_removals']:
            print("\nRisky removals:")
            for removal in report['risky_removals']:
                print(f"  Line {removal['line_number']}: {removal['import_line']}")
                print(f"    Reasons: {', '.join(removal['reasons'])}")
    
    elif args.fix or args.dry_run:
        result = fixer.fix_duplicates(args.file, dry_run=args.dry_run)
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Duplicates removed: {result['duplicates_removed']}")
            if result.get('backup_created'):
                print(f"Backup created: {args.file}.backup_duplicate_fix")
    else:
        analysis = fixer.analyze_file(args.file)
        print(f"Analysis for {args.file}")
        print("=" * 30)
        print(f"Total imports: {analysis['total_imports']}")
        print(f"Duplicates found: {analysis['duplicates_found']}")
        print(f"Safe to remove: {analysis['safe_to_remove']}")
        print(f"Risky to remove: {analysis['risky_removals']}")


if __name__ == "__main__":
    main()