from src.utils.tprint import tprint

from typing import Dict, List, Any, Optional
#!/usr/bin/env python3
"""
Intelligent Import Fixer

This module provides automatic fixing of import issues with different confidence levels:
- HIGH CONFIDENCE (95%): Auto-fix immediately
- MEDIUM CONFIDENCE (4%): Auto-fix with user confirmation
- LOW CONFIDENCE (1%): Flag for manual review only

This approach maximizes automation while maintaining safety.
"""

import ast
import os
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import shutil
from datetime import datetime
import numpy as np
import time


class ConfidenceLevel(Enum):
    """Confidence levels for automatic fixing."""
    HIGH = "high"      # 95% - Auto-fix immediately
    LOW = "low"        # 5% - Flag only, no auto-fix


class FixAction(Enum):
    """Actions to take for each issue."""
    AUTO_FIX = "auto_fix"
    FLAG_ONLY = "flag_only"
    SKIP = "skip"


@dataclass
class ImportIssue:
    """Represents an import issue with confidence assessment."""
    issue_type: str
    confidence: ConfidenceLevel
    action: FixAction
    line_number: int
    original_line: str
    suggested_fix: str
    reason: str
    safety_score: int = 0
    max_safety_score: int = 4
    can_auto_fix: bool = False
    manual_review_needed: bool = False


@dataclass
class FixResult:
    """Result of an import fix operation."""
    file_path: str
    total_issues: int
    auto_fixed: int
    flagged_only: int
    skipped: int
    backup_created: bool
    errors: List[str] = field(default_factory=list)
    fixed_lines: List[int] = field(default_factory=list)
    flagged_issues: List[ImportIssue] = field(default_factory=list)


class IntelligentImportFixer:
    """
    Intelligent import fixer that automatically handles different confidence levels.
    
    This class provides:
    1. Automatic fixing of high-confidence issues (95%)
    2. Confirmation-based fixing of medium-confidence issues (4%)
    3. Flagging of low-confidence issues (1%)
    4. Comprehensive safety analysis
    5. Backup and rollback capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent import fixer."""
        self.config = config or {}
        self.auto_fix_enabled = self.config.get('auto_fix_enabled', True)
        self.confirmation_enabled = self.config.get('confirmation_enabled', True)
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.dry_run = self.config.get('dry_run', False)
        
        # Confidence thresholds
        self.high_confidence_threshold = 3  # 3/4 safety checks pass - auto-fix
        self.low_confidence_threshold = 2   # 2/4 safety checks pass - flag only
        
        # Side effect modules (risky to duplicate)
        self.side_effect_modules = {
            'matplotlib', 'matplotlib.pyplot', 'pylab',
            'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
            'tensorflow', 'torch', 'jax',
            'numpy.random', 'random'
        }
        
        # Order-sensitive modules
        self.order_sensitive_modules = {
            'os', 'sys', 'pathlib', 'typing',
            'collections', 'itertools', 'functools'
        }
    
    def analyze_and_fix_file(self, file_path: str, interactive: bool = False) -> FixResult:
        """Analyze and fix import issues in a file with intelligent confidence assessment."""
        result = FixResult(
            file_path=file_path,
            total_issues=0,
            auto_fixed=0,
            flagged_only=0,
            skipped=0,
            backup_created=False
        )
        
        try:
            # Create backup if not dry run
            if not self.dry_run and self.backup_enabled:
                backup_path = self._create_backup(file_path)
                result.backup_created = True
            
            # Analyze the file
            issues = self._analyze_import_issues(file_path)
            result.total_issues = len(issues)
            
            # Categorize issues by confidence level
            high_confidence_issues = [i for i in issues if i.confidence == ConfidenceLevel.HIGH]
            low_confidence_issues = [i for i in issues if i.confidence == ConfidenceLevel.LOW]
            
            tprint(f"\n📊 Analysis Results for {file_path}:")
            tprint(f"   High confidence (auto-fix): {len(high_confidence_issues)}")
            tprint(f"   Low confidence (flag only): {len(low_confidence_issues)}")
            
            # Auto-fix high confidence issues
            if high_confidence_issues and self.auto_fix_enabled:
                auto_fixed = self._auto_fix_issues(file_path, high_confidence_issues)
                result.auto_fixed = auto_fixed
                result.fixed_lines.extend([i.line_number for i in high_confidence_issues[:auto_fixed]])
                tprint(f"✅ Auto-fixed {auto_fixed} high-confidence issues")
            
            # Flag low confidence issues
            if low_confidence_issues:
                result.flagged_issues.extend(low_confidence_issues)
                result.flagged_only += len(low_confidence_issues)
                tprint(f"🚩 Flagged {len(low_confidence_issues)} low-confidence issues for manual review")
            
            return result
            
        except Exception as e:
            result.errors.append(str(e))
            return result
    
    def _analyze_import_issues(self, file_path: str) -> List[ImportIssue]:
        """Analyze a file for import issues and assess confidence levels."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            tree = ast.parse(content)
            
            # Find duplicate imports
            duplicate_issues = self._find_duplicate_imports(tree, lines)
            issues.extend(duplicate_issues)
            
            # Find relative import issues
            relative_issues = self._find_relative_imports(tree, lines, file_path)
            issues.extend(relative_issues)
            
            return issues
            
        except Exception as e:
            # Add parse error as low confidence issue
            issues.append(ImportIssue(
                issue_type="parse_error",
                confidence=ConfidenceLevel.LOW,
                action=FixAction.FLAG_ONLY,
                line_number=0,
                original_line="",
                suggested_fix="Fix syntax errors before analyzing imports",
                reason=f"Parse error: {str(e)}",
                safety_score=0,
                can_auto_fix=False,
                manual_review_needed=True
            ))
            return issues
    
    def _find_duplicate_imports(self, tree: ast.AST, lines: List[str]) -> List[ImportIssue]:
        """Find duplicate imports and assess confidence for fixing."""
        issues = []
        seen_imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    effective_name = alias.asname or alias.name.split('.')[-1]
                    
                    if effective_name in seen_imports:
                        # Found duplicate
                        original = seen_imports[effective_name]
                        confidence, action, reason = self._assess_duplicate_confidence(
                            node, alias, lines, tree
                        )
                        
                        issue = ImportIssue(
                            issue_type="duplicate_import",
                            confidence=confidence,
                            action=action,
                            line_number=node.lineno,
                            original_line=lines[node.lineno - 1].strip(),
                            suggested_fix=f"Remove duplicate import: {lines[node.lineno - 1].strip()}",
                            reason=reason,
                            safety_score=self._calculate_safety_score(node, alias, lines, tree),
                            can_auto_fix=(confidence == ConfidenceLevel.HIGH),
                            manual_review_needed=(confidence == ConfidenceLevel.LOW)
                        )
                        issues.append(issue)
                    else:
                        seen_imports[effective_name] = (node, alias)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    effective_name = alias.asname or alias.name
                    
                    if effective_name in seen_imports:
                        # Found duplicate
                        original = seen_imports[effective_name]
                        confidence, action, reason = self._assess_duplicate_confidence(
                            node, alias, lines, tree
                        )
                        
                        issue = ImportIssue(
                            issue_type="duplicate_import",
                            confidence=confidence,
                            action=action,
                            line_number=node.lineno,
                            original_line=lines[node.lineno - 1].strip(),
                            suggested_fix=f"Remove duplicate import: {lines[node.lineno - 1].strip()}",
                            reason=reason,
                            safety_score=self._calculate_safety_score(node, alias, lines, tree),
                            can_auto_fix=(confidence == ConfidenceLevel.HIGH),
                            manual_review_needed=(confidence == ConfidenceLevel.LOW)
                        )
                        issues.append(issue)
                    else:
                        seen_imports[effective_name] = (node, alias)
        
        return issues
    
    def _find_relative_imports(self, tree: ast.AST, lines: List[str], file_path: str) -> List[ImportIssue]:
        """Find relative imports and assess confidence for fixing."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('.'):
                relative_levels = len(node.module) - len(node.module.lstrip('.'))
                confidence, action, reason = self._assess_relative_import_confidence(
                    node, lines, file_path, relative_levels
                )
                
                issue = ImportIssue(
                    issue_type="relative_import",
                    confidence=confidence,
                    action=action,
                    line_number=node.lineno,
                    original_line=lines[node.lineno - 1].strip(),
                    suggested_fix=self._suggest_absolute_import(node, file_path),
                    reason=reason,
                    safety_score=self._calculate_relative_safety_score(node, lines, file_path),
                    can_auto_fix=(confidence == ConfidenceLevel.HIGH),
                    requires_confirmation=(confidence == ConfidenceLevel.MEDIUM),
                    manual_review_needed=(confidence == ConfidenceLevel.LOW)
                )
                issues.append(issue)
        
        return issues
    
    def _assess_duplicate_confidence(self, node, alias, lines: List[str], tree: ast.AST) -> Tuple[ConfidenceLevel, FixAction, str]:
        """Assess confidence level for fixing a duplicate import."""
        safety_score = self._calculate_safety_score(node, alias, lines, tree)
        
        if safety_score >= self.high_confidence_threshold:
            return ConfidenceLevel.HIGH, FixAction.AUTO_FIX, "Safe to remove - no side effects or dependencies detected"
        else:
            return ConfidenceLevel.LOW, FixAction.FLAG_ONLY, "Risky - has side effects, dependencies, or dynamic usage"
    
    def _assess_relative_import_confidence(self, node, lines: List[str], file_path: str, relative_levels: int) -> Tuple[ConfidenceLevel, FixAction, str]:
        """Assess confidence level for fixing a relative import."""
        safety_score = self._calculate_relative_safety_score(node, lines, file_path)
        
        if self._is_standalone_script(file_path):
            return ConfidenceLevel.HIGH, FixAction.AUTO_FIX, "Standalone script - convert to absolute import"
        elif safety_score >= self.high_confidence_threshold:
            return ConfidenceLevel.HIGH, FixAction.AUTO_FIX, "Safe to convert to absolute import"
        else:
            return ConfidenceLevel.LOW, FixAction.FLAG_ONLY, "Complex relative import - manual review needed"
    
    def _calculate_safety_score(self, node, alias, lines: List[str], tree: ast.AST) -> int:
        """Calculate safety score for duplicate import removal (0-4)."""
        score = 0
        
        # Check 1: No side effects
        if not self._has_side_effects(alias.name, node):
            score += 1
        
        # Check 2: Not in conditional block
        if not self._is_conditional_import(lines, node.lineno):
            score += 1
        
        # Check 3: No dynamic access
        if not self._has_dynamic_access(tree, alias.name, node.lineno):
            score += 1
        
        # Check 4: Import order not significant
        if not self._has_import_order_dependencies(alias.name):
            score += 1
        
        return score
    
    def _calculate_relative_safety_score(self, node, lines: List[str], file_path: str) -> int:
        """Calculate safety score for relative import conversion (0-4)."""
        score = 0
        
        # Check 1: Not in standalone script
        if not self._is_standalone_script(file_path):
            score += 1
        
        # Check 2: Not in test file
        if not self._is_test_file(file_path):
            score += 1
        
        # Check 3: Not in conditional block
        if not self._is_conditional_import(lines, node.lineno):
            score += 1
        
        # Check 4: Simple relative import
        if not self._is_complex_relative_import(node):
            score += 1
        
        return score
    
    def _has_side_effects(self, module_name: str, node) -> bool:
        """Check if module has side effects when imported."""
        module_parts = module_name.split('.')
        for i in range(len(module_parts)):
            partial_module = '.'.join(module_parts[:i+1])
            if partial_module in self.side_effect_modules:
                return True
        return False
    
    def _is_conditional_import(self, lines: List[str], line_number: int) -> bool:
        """Check if import is in a conditional block."""
        if line_number <= 1:
            return False
        
        line = lines[line_number - 1]
        return (line.strip().startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'with ')) or
                line.startswith(('    ', '\t')))
    
    def _has_dynamic_access(self, tree: ast.AST, name: str, line_number: int) -> bool:
        """Check if import has dynamic access patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                if node.lineno > line_number:
                    # Check if it's used in dynamic context
                    for parent in ast.walk(tree):
                        if (isinstance(parent, (ast.Call, ast.Attribute)) and 
                            hasattr(parent, 'lineno') and parent.lineno == node.lineno):
                            if (isinstance(parent, ast.Call) and 
                                isinstance(parent.func, ast.Name) and 
                                parent.func.id in ['globals', 'locals', 'getattr', 'setattr']):
                                return True
        return False
    
    def _has_import_order_dependencies(self, module_name: str) -> bool:
        """Check if import order is significant."""
        return module_name in self.order_sensitive_modules
    
    def _is_standalone_script(self, file_path: str) -> bool:
        """Check if file is a standalone script."""
        path = Path(file_path)
        return (path.name == '__main__.py' or 
                not any((path.parent / '__init__.py').exists() for _ in range(3)))
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        path = Path(file_path)
        return ('test' in path.name.lower() or 
                'test' in path.parts or
                path.name.startswith('test_'))
    
    def _is_complex_relative_import(self, node) -> bool:
        """Check if relative import is complex."""
        if isinstance(node, ast.ImportFrom):
            return len(node.names) > 3 or any(alias.name == '*' for alias in node.names)
        return False
    
    def _suggest_absolute_import(self, node, file_path: str) -> str:
        """Suggest absolute import replacement."""
        if isinstance(node, ast.ImportFrom):
            # Try to determine package name from file path
            path = Path(file_path)
            package_parts = []
            
            # Walk up directory tree looking for __init__.py
            current = path.parent
            while current != current.parent:
                if (current / '__init__.py').exists():
                    package_parts.insert(0, current.name)
                current = current.parent
                if len(package_parts) >= 2:  # Limit depth
                    break
            
            if package_parts:
                package_name = '.'.join(package_parts)
                module = node.module.lstrip('.')
                if module:
                    return f"from {package_name}.{module} import {', '.join(alias.name for alias in node.names)}"
                else:
                    return f"from {package_name} import {', '.join(alias.name for alias in node.names)}"
        
        return "Convert to absolute import"
    
    def _auto_fix_issues(self, file_path: str, issues: List[ImportIssue]) -> int:
        """Automatically fix high-confidence issues."""
        if self.dry_run:
            tprint(f"DRY RUN: Would auto-fix {len(issues)} issues")
            return len(issues)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Remove lines in reverse order to maintain line numbers
            lines_to_remove = sorted([i.line_number - 1 for i in issues], reverse=True)
            
            for line_num in lines_to_remove:
                lines.pop(line_num)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return len(issues)
            
        except Exception as e:
            tprint(f"Error auto-fixing issues: {e}")
            return 0
    
    
    def _fix_single_issue(self, file_path: str, issue: ImportIssue) -> bool:
        """Fix a single import issue."""
        if self.dry_run:
            tprint(f"DRY RUN: Would fix {issue.issue_type} at line {issue.line_number}")
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if issue.issue_type == "duplicate_import":
                # Remove the duplicate line
                lines.pop(issue.line_number - 1)
            elif issue.issue_type == "relative_import":
                # Replace with absolute import
                lines[issue.line_number - 1] = issue.suggested_fix + '\n'
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True
            
        except Exception as e:
            tprint(f"Error fixing issue: {e}")
            return False
    
    def _create_backup(self, file_path: str) -> str:
        """Create backup of file before making changes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def generate_report(self, results: List[FixResult]) -> Dict[str, Any]:
        """Generate comprehensive report of all fix operations."""
        total_files = len(results)
        total_issues = sum(r.total_issues for r in results)
        total_auto_fixed = sum(r.auto_fixed for r in results)
        total_flagged = sum(r.flagged_only for r in results)
        
        return {
            "summary": {
                "total_files_processed": total_files,
                "total_issues_found": total_issues,
                "auto_fixed": total_auto_fixed,
                "flagged_for_review": total_flagged,
                "auto_fix_rate": (total_auto_fixed / total_issues * 100) if total_issues > 0 else 0,
                "total_fix_rate": (total_auto_fixed / total_issues * 100) if total_issues > 0 else 0
            },
            "files": [
                {
                    "file_path": r.file_path,
                    "issues_found": r.total_issues,
                    "auto_fixed": r.auto_fixed,
                    "flagged": r.flagged_only,
                    "backup_created": r.backup_created,
                    "errors": r.errors
                }
                for r in results
            ],
            "flagged_issues": [
                {
                    "file_path": r.file_path,
                    "issue_type": issue.issue_type,
                    "confidence": issue.confidence.value,
                    "line_number": issue.line_number,
                    "reason": issue.reason,
                    "suggested_fix": issue.suggested_fix
                }
                for r in results
                for issue in r.flagged_issues
            ]
        }


def main():
    """Command-line interface for intelligent import fixing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Import Fixer")
    parser.add_argument("target", help="File or directory to fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--no-auto-fix", action="store_true", help="Disable automatic fixing")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")
    parser.add_argument("--output", "-o", help="Output report file")
    
    args = parser.parse_args()
    
    config = {
        'auto_fix_enabled': not args.no_auto_fix,
        'backup_enabled': not args.no_backup,
        'dry_run': args.dry_run
    }
    
    fixer = IntelligentImportFixer(config)
    results = []
    
    target_path = Path(args.target)
    
    if target_path.is_file():
        # Single file
        result = fixer.analyze_and_fix_file(str(target_path), interactive=False)
        results.append(result)
    else:
        # Directory
        for py_file in target_path.rglob("*.py"):
            result = fixer.analyze_and_fix_file(str(py_file), interactive=False)
            results.append(result)
    
    # Generate and display report
    report = fixer.generate_report(results)
    
    tprint(f"\n📊 INTELLIGENT IMPORT FIXING REPORT")
    tprint("=" * 50)
    tprint(f"Files processed: {report['summary']['total_files_processed']}")
    tprint(f"Total issues found: {report['summary']['total_issues_found']}")
    tprint(f"Auto-fixed: {report['summary']['auto_fixed']} ({report['summary']['auto_fix_rate']:.1f}%)")
    tprint(f"Flagged for review: {report['summary']['flagged_for_review']}")
    tprint(f"Total fix rate: {report['summary']['total_fix_rate']:.1f}%")
    
    if report['flagged_issues']:
        tprint(f"\n🚩 Issues flagged for manual review:")
        for issue in report['flagged_issues']:
            tprint(f"  {issue['file_path']}:{issue['line_number']} - {issue['reason']}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        tprint(f"\n💾 Report saved to: {args.output}")


if __name__ == "__main__":
    main()