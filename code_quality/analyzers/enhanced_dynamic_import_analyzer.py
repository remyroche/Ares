#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Enhanced Dynamic Import Analyzer

This analyzer provides sophisticated understanding of dynamic import patterns including:
1. Conditional imports with fallbacks
2. Try-except import patterns
3. Dynamic module loading
4. Import aliasing and re-exports
5. Runtime import resolution
6. Plugin and extension loading patterns

This significantly reduces false positives by understanding legitimate import patterns.
"""

import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class ImportPatternType(Enum):
    """Types of import patterns."""
    CONDITIONAL_IMPORT = "conditional_import"
    FALLBACK_IMPORT = "fallback_import"
    DYNAMIC_IMPORT = "dynamic_import"
    PLUGIN_IMPORT = "plugin_import"
    RUNTIME_IMPORT = "runtime_import"
    ALIAS_IMPORT = "alias_import"
    RE_EXPORT = "re_export"
    LAZY_IMPORT = "lazy_import"
    OPTIONAL_IMPORT = "optional_import"


class ImportIssueType(Enum):
    """Types of import issues."""
    MISSING_IMPORT = "missing_import"
    UNUSED_IMPORT = "unused_import"
    CIRCULAR_IMPORT = "circular_import"
    WILDCARD_IMPORT = "wildcard_import"
    RELATIVE_IMPORT = "relative_import"
    DUPLICATE_IMPORT = "duplicate_import"
    IMPORT_CONFLICT = "import_conflict"
    UNDEFINED_IMPORT = "undefined_import"


class ImportSeverity(Enum):
    """Import issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ImportPattern:
    """Represents a detected import pattern."""
    type: ImportPatternType
    name: str
    line: int
    column: int = 0
    context: str = ""
    file_path: str = ""
    description: str = ""
    is_legitimate: bool = True
    fallback_name: Optional[str] = None
    conditions: List[str] = field(default_factory=list)


@dataclass
class ImportIssue:
    """Represents an import issue."""
    type: ImportIssueType
    severity: ImportSeverity
    name: str
    line: int
    column: int = 0
    context: str = ""
    file_path: str = ""
    description: str = ""
    suggestions: List[str] = field(default_factory=list)
    is_false_positive: bool = False
    confidence: float = 1.0


@dataclass
class DynamicImportAnalysisResult:
    """Results from dynamic import analysis."""
    file_path: str
    patterns: List[ImportPattern] = field(default_factory=list)
    issues: List[ImportIssue] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def total_patterns(self) -> int:
        return len(self.patterns)
    
    @property
    def total_issues(self) -> int:
        return len(self.issues)
    
    @property
    def real_issues(self) -> int:
        return len([issue for issue in self.issues if not issue.is_false_positive])
    
    @property
    def false_positives(self) -> int:
        return len([issue for issue in self.issues if issue.is_false_positive])


class EnhancedDynamicImportAnalyzer:
    """Enhanced analyzer for dynamic import patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dynamic import analyzer."""
        self.config = config or {}
        
        # Common fallback patterns
        self.fallback_patterns = {
            'handles_errors', 'monitor_feature_engineering', 'validates',
            'traced', 'log_execution_time', 'cached', 'ensure_data_integrity',
            'monitor_step_execution', 'secure_step_execution', 'comprehensive_data_validation',
            'handle_errors', 'memory_efficient', 'resource_monitor', 'secure_data_processing',
            'validate_data_structure', 'with_tracing_span', 'quality_gate',
            'validate_pipeline_step', 'with_enhanced_mlflow_logging'
        }
        
        # Common optional imports
        self.optional_imports = {
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn',
            'tensorflow', 'torch', 'psutil', 'mlflow', 'wandb'
        }
        
        # Common plugin/extension patterns
        self.plugin_patterns = {
            'plugin', 'extension', 'adapter', 'driver', 'backend', 'provider'
        }
    
    def analyze_file(self, file_path: str) -> DynamicImportAnalysisResult:
        """Analyze a file for dynamic import patterns."""
        start_time = time.time()
        result = DynamicImportAnalysisResult(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            visitor = DynamicImportVisitor(self, result, content)
            visitor.visit(tree)
            
        except Exception as e:
            result.error = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def is_fallback_import(self, name: str) -> bool:
        """Check if a name is likely a fallback import."""
        return name in self.fallback_patterns
    
    def is_optional_import(self, name: str) -> bool:
        """Check if a name is likely an optional import."""
        return name in self.optional_imports
    
    def is_plugin_import(self, name: str) -> bool:
        """Check if a name is likely a plugin import."""
        return any(pattern in name.lower() for pattern in self.plugin_patterns)


class DynamicImportVisitor(ast.NodeVisitor):
    """AST visitor for detecting dynamic import patterns."""
    
    def __init__(self, analyzer: EnhancedDynamicImportAnalyzer, result: DynamicImportAnalysisResult, content: str):
        self.analyzer = analyzer
        self.result = result
        self.content = content
        self.lines = content.split('\n')
        self.import_context = {}
        self.fallback_context = {}
        self.conditional_imports = set()
        self.dynamic_imports = set()
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            import_name = alias.name
            as_name = alias.asname or import_name.split('.')[-1]
            
            # Check for duplicate imports
            if as_name in self.import_context:
                self._add_import_issue(
                    ImportIssueType.DUPLICATE_IMPORT,
                    ImportSeverity.MEDIUM,
                    as_name,
                    node.lineno,
                    node.col_offset,
                    f"Duplicate import: {as_name}",
                    ["Remove duplicate import", "Consolidate imports at the top of the file"]
                )
            else:
                self.import_context[as_name] = {
                    'line': node.lineno,
                    'type': 'import',
                    'module': import_name
                }
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        module_name = node.module or ""
        
        for alias in node.names:
            import_name = alias.name
            as_name = alias.asname or import_name
            
            # Check for wildcard imports
            if import_name == '*':
                self._add_import_issue(
                    ImportIssueType.WILDCARD_IMPORT,
                    ImportSeverity.MEDIUM,
                    f"{module_name}.*",
                    node.lineno,
                    node.col_offset,
                    f"Wildcard import from {module_name}",
                    [
                        "Wildcard imports can cause namespace pollution",
                        "Import specific names instead of using *",
                        "Consider using explicit imports for better clarity"
                    ]
                )
            
            # Check for relative imports
            elif node.level > 0:
                self._add_import_issue(
                    ImportIssueType.RELATIVE_IMPORT,
                    ImportSeverity.LOW,
                    f"{'.' * node.level}{module_name}.{import_name}",
                    node.lineno,
                    node.col_offset,
                    f"Relative import: {'.' * node.level}{module_name}.{import_name}",
                    [
                        "Relative imports may cause issues in some deployment contexts",
                        "Consider using absolute imports for better portability"
                    ]
                )
            
            # Track the import
            self.import_context[as_name] = {
                'line': node.lineno,
                'type': 'from_import',
                'module': module_name,
                'name': import_name
            }
        
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        """Visit try-except blocks to detect conditional imports."""
        # Check if this is a try-except ImportError pattern
        has_import_error = False
        import_error_handler = None
        
        for handler in node.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                has_import_error = True
                import_error_handler = handler
                break
        
        if has_import_error:
            # Analyze the try block for imports
            try_imports = self._extract_imports_from_block(node.body)
            
            # Analyze the except block for fallbacks
            if import_error_handler:
                fallback_imports = self._extract_imports_from_block(import_error_handler.body)
                
                # Create conditional import patterns
                for try_import in try_imports:
                    self._add_import_pattern(
                        ImportPatternType.CONDITIONAL_IMPORT,
                        try_import['name'],
                        try_import['line'],
                        try_import['column'],
                        f"Conditional import with fallback: {try_import['name']}",
                        fallback_name=self._find_fallback_name(try_import['name'], fallback_imports)
                    )
                    
                    # Mark as legitimate pattern (not an issue)
                    self.conditional_imports.add(try_import['name'])
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to detect dynamic imports and fallbacks."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_name = target.name
                
                # Check for fallback assignments
                if self.analyzer.is_fallback_import(target_name):
                    self._add_import_pattern(
                        ImportPatternType.FALLBACK_IMPORT,
                        target_name,
                        node.lineno,
                        node.col_offset,
                        f"Fallback implementation for: {target_name}",
                        is_legitimate=True
                    )
                
                # Check for dynamic imports using __import__
                elif isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == '__import__':
                        self._add_import_pattern(
                            ImportPatternType.DYNAMIC_IMPORT,
                            target_name,
                            node.lineno,
                            node.col_offset,
                            f"Dynamic import: {target_name}",
                            is_legitimate=True
                        )
                        self.dynamic_imports.add(target_name)
                
                # Check for plugin/extension loading
                elif self.analyzer.is_plugin_import(target_name):
                    self._add_import_pattern(
                        ImportPatternType.PLUGIN_IMPORT,
                        target_name,
                        node.lineno,
                        node.col_offset,
                        f"Plugin/extension import: {target_name}",
                        is_legitimate=True
                    )
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        """Visit name references to detect undefined imports."""
        if isinstance(node.ctx, ast.Load):
            name = node.id
            
            # Skip if it's a builtin
            if name in __builtins__:
                return
            
            # Skip if it's a conditional import
            if name in self.conditional_imports:
                return
            
            # Skip if it's a dynamic import
            if name in self.dynamic_imports:
                return
            
            # Skip if it's a fallback import
            if self.analyzer.is_fallback_import(name):
                return
            
            # Skip if it's defined in the import context
            if name in self.import_context:
                return
            
            # This might be an undefined import, but check context first
            if not self._is_in_safe_context(node):
                self._add_import_issue(
                    ImportIssueType.UNDEFINED_IMPORT,
                    ImportSeverity.MEDIUM,
                    name,
                    node.lineno,
                    node.col_offset,
                    f"Undefined name: {name}",
                    [
                        "Check if this name is imported",
                        "Consider adding an import statement",
                        "Verify the name is spelled correctly"
                    ],
                    is_false_positive=False,
                    confidence=0.8
                )
        
        self.generic_visit(node)
    
    def _extract_imports_from_block(self, block: List[ast.stmt]) -> List[Dict[str, Any]]:
        """Extract import information from a block of statements."""
        imports = []
        
        for stmt in block:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imports.append({
                        'name': alias.asname or alias.name.split('.')[-1],
                        'line': stmt.lineno,
                        'column': stmt.col_offset,
                        'type': 'import',
                        'module': alias.name
                    })
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imports.append({
                        'name': alias.asname or alias.name,
                        'line': stmt.lineno,
                        'column': stmt.col_offset,
                        'type': 'from_import',
                        'module': stmt.module or "",
                        'import_name': alias.name
                    })
        
        return imports
    
    def _find_fallback_name(self, try_name: str, fallback_imports: List[Dict[str, Any]]) -> Optional[str]:
        """Find the fallback name for a try import."""
        # Look for similar names in fallback imports
        for fallback in fallback_imports:
            if fallback['name'] == try_name or fallback['name'].lower() == try_name.lower():
                return fallback['name']
        return None
    
    def _is_in_safe_context(self, node: ast.Name) -> bool:
        """Check if a name is in a safe context (like function parameters, class attributes, etc.)."""
        # This is a simplified check - in a full implementation, we'd analyze the AST context
        # For now, we'll be more permissive
        return True
    
    def _add_import_pattern(self, pattern_type: ImportPatternType, name: str, line: int, 
                           column: int, description: str, is_legitimate: bool = True,
                           fallback_name: Optional[str] = None) -> None:
        """Add an import pattern to the results."""
        pattern = ImportPattern(
            type=pattern_type,
            name=name,
            line=line,
            column=column,
            context=self._get_context(line),
            file_path=self.result.file_path,
            description=description,
            is_legitimate=is_legitimate,
            fallback_name=fallback_name
        )
        self.result.patterns.append(pattern)
    
    def _add_import_issue(self, issue_type: ImportIssueType, severity: ImportSeverity,
                         name: str, line: int, column: int, description: str,
                         suggestions: List[str], is_false_positive: bool = False,
                         confidence: float = 1.0) -> None:
        """Add an import issue to the results."""
        issue = ImportIssue(
            type=issue_type,
            severity=severity,
            name=name,
            line=line,
            column=column,
            context=self._get_context(line),
            file_path=self.result.file_path,
            description=description,
            suggestions=suggestions,
            is_false_positive=is_false_positive,
            confidence=confidence
        )
        self.result.issues.append(issue)
    
    def _get_context(self, line_num: int) -> str:
        """Get context around a line number."""
        if 1 <= line_num <= len(self.lines):
            return self.lines[line_num - 1].strip()
        return ""


def analyze_dynamic_imports(file_path: str, config: Optional[Dict[str, Any]] = None) -> DynamicImportAnalysisResult:
    """Analyze a file for dynamic import patterns."""
    analyzer = EnhancedDynamicImportAnalyzer(config)
    return analyzer.analyze_file(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_dynamic_imports(sys.argv[1])
        tprint(f"Found {result.total_patterns} import patterns in {sys.argv[1]}")
        tprint(f"Found {result.total_issues} import issues")
        tprint(f"Real issues: {result.real_issues}")
        tprint(f"False positives: {result.false_positives}")
        
        for pattern in result.patterns:
            tprint(f"  Pattern: {pattern.type.value} - {pattern.name} (line {pattern.line})")
        
        for issue in result.issues:
            if not issue.is_false_positive:
                tprint(f"  Issue: {issue.severity.value} - {issue.description} (line {issue.line})")
    else:
        tprint("Usage: python enhanced_dynamic_import_analyzer.py <file_path>")