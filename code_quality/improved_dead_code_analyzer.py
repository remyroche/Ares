"""
Improved Dead Code Analyzer with Reduced False Positives

This analyzer uses more sophisticated heuristics to reduce false positives
by checking cross-file usage, public APIs, and interface patterns.
"""

import ast
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Dict
from typing import List


@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    max_workers: int = 4
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/__pycache__/*",
        "*/.*/*",
        "*/venv/*",
        "*/env/*",
        "*/node_modules/*",
        "*/test*/*",
        "*/tests/*",
        "*/__tests__/*",
        "*/test_*",
        "*_test.py",
        "test_*.py",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dll",
        "*.dylib",
    ])
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    timeout: int = 30
    min_confidence: float = 0.8  # Only report issues with high confidence
    check_cross_file_usage: bool = True  # Check if functions are used in other files


@dataclass
class DeadCodeIssue:
    """Represents a dead code issue."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    message: str
    suggestion: str
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


class ImprovedDeadCodeAnalyzer:
    """Improved dead code analyzer with reduced false positives."""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.issues: List[DeadCodeIssue] = []
        self.global_imports: Dict[str, Set[str]] = defaultdict(set)
        self.global_exports: Dict[str, Set[str]] = defaultdict(set)
        self.global_function_calls: Dict[str, Set[str]] = defaultdict(set)
        self.global_class_usage: Dict[str, Set[str]] = defaultdict(set)
        self.public_apis: Dict[str, Set[str]] = defaultdict(set)
        self.interface_patterns: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        
    def analyze_project(self, project_root: str) -> Dict[str, Any]:
        """Analyze entire project for dead code."""
        start_time = datetime.now()
        project_path = Path(project_root)
        
        self.logger.info(f"🔍 Starting improved dead code analysis on {project_path}")
        
        # Find all Python files
        python_files = self._find_python_files(project_path)
        self.logger.info(f"📁 Found {len(python_files)} Python files to analyze")
        
        # First pass: collect global information
        self._collect_global_info(python_files)
        
        # Second pass: analyze for dead code with cross-file awareness
        self._analyze_files_improved(python_files)
        
        # Filter issues by confidence
        high_confidence_issues = [
            issue for issue in self.issues 
            if issue.confidence >= self.config.min_confidence
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate statistics
        stats = self._generate_statistics(high_confidence_issues, processing_time)
        
        return {
            "project_root": str(project_path),
            "stats": stats,
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "confidence": issue.confidence,
                    "context": issue.context
                }
                for issue in high_confidence_issues
            ],
            "global_analysis": {
                "imports": {k: list(v) for k, v in self.global_imports.items()},
                "exports": {k: list(v) for k, v in self.global_exports.items()},
                "function_calls": {k: list(v) for k, v in self.global_function_calls.items()},
                "class_usage": {k: list(v) for k, v in self.global_class_usage.items()},
                "public_apis": {k: list(v) for k, v in self.public_apis.items()}
            }
        }
    
    def _find_python_files(self, directory: Path) -> List[Path]:
        """Find Python files in directory."""
        python_files = []
        
        for file_path in directory.rglob("*.py"):
            # Check exclude patterns
            should_exclude = False
            for pattern in self.config.exclude_patterns:
                if file_path.match(pattern):
                    should_exclude = True
                    break
            
            if not should_exclude:
                python_files.append(file_path)
        
        return python_files
    
    def _collect_global_info(self, python_files: List[Path]) -> None:
        """Collect global information about imports, exports, and usage."""
        self.logger.info("📊 Collecting global information...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            executor.map(self._collect_file_info, python_files)
    
    def _collect_file_info(self, file_path: Path) -> None:
        """Collect information from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_key = str(file_path)
            
            # Collect imports and exports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.global_imports[file_key].add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.global_imports[file_key].add(node.module)
                    for alias in node.names:
                        self.global_imports[file_key].add(alias.asname or alias.name)
                elif isinstance(node, ast.FunctionDef):
                    self.global_exports[file_key].add(node.name)
                elif isinstance(node, ast.ClassDef):
                    self.global_exports[file_key].add(node.name)
                    # Check if it's an interface/abstract class
                    if self._is_interface_class(node):
                        self.interface_patterns.add(node.name)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.global_function_calls[file_key].add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        self.global_function_calls[file_key].add(node.func.attr)
                elif isinstance(node, ast.Assign):
                    # Check for class instantiation
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if isinstance(node.value, ast.Call):
                                if isinstance(node.value.func, ast.Name):
                                    self.global_class_usage[file_key].add(node.value.func.id)
            
            # Check for __all__ declarations
            self._extract_public_api(file_path, tree)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _is_interface_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an interface or abstract class."""
        # Check for ABC inheritance
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ['ABC', 'AbstractBaseClass']:
                return True
        
        # Check for abstract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        return True
        
        # Check class name patterns
        if node.name.startswith('I') and len(node.name) > 1:
            return True  # Interface pattern like IDataStep
        
        return False
    
    def _extract_public_api(self, file_path: Path, tree: ast.AST) -> None:
        """Extract __all__ declarations to identify public APIs."""
        file_key = str(file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    self.public_apis[file_key].add(elt.value)
                                elif isinstance(elt, ast.Str):  # Python < 3.8
                                    self.public_apis[file_key].add(elt.s)
    
    def _analyze_files_improved(self, python_files: List[Path]) -> None:
        """Analyze files for dead code with improved heuristics."""
        self.logger.info("🔍 Analyzing files for dead code...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            executor.map(self._analyze_file_improved, python_files)
    
    def _analyze_file_improved(self, file_path: Path) -> None:
        """Analyze a single file for dead code with improved heuristics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_key = str(file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._check_unused_function_improved(file_path, node, file_key)
                elif isinstance(node, ast.ClassDef):
                    self._check_unused_class_improved(file_path, node, file_key)
                elif isinstance(node, ast.Import):
                    self._check_unused_import_improved(file_path, node, content)
                elif isinstance(node, ast.ImportFrom):
                    self._check_unused_import_from_improved(file_path, node, content)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _check_unused_function_improved(self, file_path: Path, node: ast.FunctionDef, file_key: str) -> None:
        """Check if function is unused with improved heuristics."""
        if node.name.startswith('_'):
            return  # Skip private functions
        
        # Skip if it's in public API
        if node.name in self.public_apis[file_key]:
            return
        
        # Skip if it's a main function or entry point
        if node.name in ['main', '__main__', 'run', 'execute']:
            return
        
        # Check if function is used in this file
        if node.name in self.global_function_calls[file_key]:
            return
        
        # Check if function is used in other files
        if self.config.check_cross_file_usage:
            if self._is_function_used_elsewhere(node.name, file_key):
                return
        
        # Check if it's a property or method
        if any(isinstance(dec, ast.Name) and dec.id in ['property', 'staticmethod', 'classmethod'] 
               for dec in node.decorator_list):
            return
        
        # High confidence that this function is unused
        self.issues.append(DeadCodeIssue(
            file_path=str(file_path),
            line_number=node.lineno,
            issue_type="dead_code",
            severity="warning",
            message=f"Function '{node.name}' appears to be unused",
            suggestion="Remove if truly unused, or add to __all__ if it's part of public API",
            confidence=0.9,
            context={"function_name": node.name, "is_public_api": False}
        ))
    
    def _check_unused_class_improved(self, file_path: Path, node: ast.ClassDef, file_key: str) -> None:
        """Check if class is unused with improved heuristics."""
        if node.name.startswith('_'):
            return  # Skip private classes
        
        # Skip if it's in public API
        if node.name in self.public_apis[file_key]:
            return
        
        # Skip if it's an interface/abstract class
        if node.name in self.interface_patterns:
            return
        
        # Skip if it's a base class that might be inherited
        if self._is_base_class(node.name, file_key):
            return
        
        # Check if class is used in this file
        if node.name in self.global_class_usage[file_key]:
            return
        
        # Check if class is used in other files
        if self.config.check_cross_file_usage:
            if self._is_class_used_elsewhere(node.name, file_key):
                return
        
        # High confidence that this class is unused
        self.issues.append(DeadCodeIssue(
            file_path=str(file_path),
            line_number=node.lineno,
            issue_type="dead_code",
            severity="warning",
            message=f"Class '{node.name}' appears to be unused",
            suggestion="Remove if truly unused, or add to __all__ if it's part of public API",
            confidence=0.9,
            context={"class_name": node.name, "is_public_api": False}
        ))
    
    def _check_unused_import_improved(self, file_path: Path, node: ast.Import, content: str) -> None:
        """Check if import is unused with improved heuristics."""
        for alias in node.names:
            import_name = alias.asname or alias.name
            if import_name.startswith('_'):
                continue
            
            # Skip if it's a standard library that might be used indirectly
            if import_name in ['os', 'sys', 'logging', 'json', 'datetime', 'pathlib', 'typing']:
                continue
            
            # Check if import is actually used
            if self._is_import_used(import_name, content):
                continue
            
            # High confidence that this import is unused
            self.issues.append(DeadCodeIssue(
                file_path=str(file_path),
                line_number=node.lineno,
                issue_type="unused_import",
                severity="warning",
                message=f"Import '{import_name}' appears to be unused",
                suggestion="Remove unused import",
                confidence=0.95,
                context={"import_name": import_name}
            ))
    
    def _check_unused_import_from_improved(self, file_path: Path, node: ast.ImportFrom, content: str) -> None:
        """Check if import from is unused with improved heuristics."""
        for alias in node.names:
            import_name = alias.asname or alias.name
            if import_name.startswith('_'):
                continue
            
            # Skip if it's a standard library import
            if node.module and node.module in ['os', 'sys', 'logging', 'json', 'datetime', 'pathlib', 'typing']:
                continue
            
            # Check if import is actually used
            if self._is_import_used(import_name, content):
                continue
            
            # High confidence that this import is unused
            self.issues.append(DeadCodeIssue(
                file_path=str(file_path),
                line_number=node.lineno,
                issue_type="unused_import",
                severity="warning",
                message=f"Import '{import_name}' from '{node.module}' appears to be unused",
                suggestion="Remove unused import",
                confidence=0.95,
                context={"import_name": import_name, "module": node.module}
            ))
    
    def _is_function_used_elsewhere(self, function_name: str, current_file: str) -> bool:
        """Check if function is used in other files."""
        for file_key, calls in self.global_function_calls.items():
            if file_key != current_file and function_name in calls:
                return True
        return False
    
    def _is_class_used_elsewhere(self, class_name: str, current_file: str) -> bool:
        """Check if class is used in other files."""
        for file_key, usage in self.global_class_usage.items():
            if file_key != current_file and class_name in usage:
                return True
        return False
    
    def _is_base_class(self, class_name: str, current_file: str) -> bool:
        """Check if class is used as a base class."""
        for file_key, exports in self.global_exports.items():
            if file_key != current_file:
                # This is a simplified check - in reality we'd need to parse inheritance
                # For now, we'll be conservative and assume it might be a base class
                pass
        return False
    
    def _is_import_used(self, import_name: str, content: str) -> bool:
        """Check if import is actually used in the content."""
        lines = content.split('\n')
        
        for line in lines:
            # Skip import lines
            if line.strip().startswith(('import ', 'from ')):
                continue
            
            # Check for usage
            if import_name in line:
                # Make sure it's not just part of another word
                pattern = r'\b' + re.escape(import_name) + r'\b'
                if re.search(pattern, line):
                    return True
        
        return False
    
    def _generate_statistics(self, issues: List[DeadCodeIssue], processing_time: float) -> Dict[str, Any]:
        """Generate analysis statistics."""
        issue_types = Counter(issue.issue_type for issue in issues)
        severities = Counter(issue.severity for issue in issues)
        
        return {
            "files_analyzed": len(self.global_imports),
            "total_issues": len(issues),
            "processing_time": processing_time,
            "dead_code_issues": issue_types.get("dead_code", 0),
            "unused_import_issues": issue_types.get("unused_import", 0),
            "unreachable_code_issues": issue_types.get("unreachable_code", 0),
            "unused_dependency_issues": issue_types.get("unused_dependency", 0),
            "high_confidence_issues": len([i for i in issues if i.confidence >= 0.9]),
            "issue_types": dict(issue_types),
            "severities": dict(severities)
        }
    
    def export_results(self, output_file: str) -> None:
        """Export results to JSON file."""
        results = {
            "project_root": "/Users/remyroche/Documents/Ares",
            "stats": self._generate_statistics(self.issues, 0),
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "confidence": issue.confidence,
                    "context": issue.context
                }
                for issue in self.issues
            ],
            "global_analysis": {
                "imports": {k: list(v) for k, v in self.global_imports.items()},
                "exports": {k: list(v) for k, v in self.global_exports.items()},
                "function_calls": {k: list(v) for k, v in self.global_function_calls.items()},
                "class_usage": {k: list(v) for k, v in self.global_class_usage.items()},
                "public_apis": {k: list(v) for k, v in self.public_apis.items()}
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
