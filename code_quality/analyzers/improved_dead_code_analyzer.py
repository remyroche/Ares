"""
Improved Dead Code Analyzer for Pipeline Integration

This analyzer integrates the enhanced dead code detection logic with reduced false positives
into the code quality pipeline. It provides comprehensive dead code analysis with
confidence scoring and cross-file usage detection.

Features:
- Public API detection using __all__ declarations
- Cross-file usage analysis
- Abstract/interface class detection
- Improved import usage detection with AST analysis
- Confidence scoring for better decision making
- Integration with the unified pipeline system
"""

import ast
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Simple config class for standalone usage
class AnalysisConfig:
    def __init__(self):
        self.project_root = None
        self.output_dir = None
        self.verbose = False

def find_python_files(directory: Path) -> list[Path]:
    """Find all Python files in a directory."""
    python_files = []
    for py_file in directory.rglob("*.py"):
        # Skip __pycache__ and other common exclusions
        if "__pycache__" not in str(py_file) and ".pyc" not in str(py_file):
            python_files.append(py_file)
    return python_files


@dataclass
class DeadCodeIssue:
    """Container for dead code analysis results."""
    file_path: str
    line_number: int
    issue_type: str  # "unused_function", "unused_class", "unused_import", "unused_import_from"
    name: str
    description: str
    confidence: float  # 0.0 to 1.0
    severity: str  # "low", "medium", "high", "critical"
    code_snippet: str = ""
    is_public_api: bool = False
    is_used_cross_file: bool = False
    is_abstract_interface: bool = False
    removal_impact: str = "low"  # low, medium, high
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DeadCodeAnalysisResult:
    """Container for complete dead code analysis results."""
    issues: List[DeadCodeIssue]
    summary: Dict[str, Any]
    file_analysis: Dict[str, Dict[str, Any]]
    global_analysis: Dict[str, Any]
    execution_time: float
    files_analyzed: int
    total_issues: int
    issues_by_type: Dict[str, int]
    issues_by_confidence: Dict[str, int]
    issues_by_severity: Dict[str, int]


class ImprovedDeadCodeAnalyzer:
    """
    Enhanced dead code analyzer with reduced false positives.
    
    This analyzer provides sophisticated dead code detection that considers:
    - Public API exposure (__all__ declarations)
    - Cross-file usage patterns
    - Abstract/interface class detection
    - Import usage patterns
    - Confidence scoring for better decision making
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the analyzer with configuration."""
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis state
        self.project_root: Optional[Path] = None
        self.python_files: List[Path] = []
        self.ast_trees: Dict[str, ast.AST] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.usage_map: Dict[str, Set[str]] = defaultdict(set)
        self.public_apis: Dict[str, Set[str]] = defaultdict(set)
        
        # Results
        self.issues: List[DeadCodeIssue] = []
        self.file_analysis: Dict[str, Dict[str, Any]] = {}
        
    def analyze_directory(self, directory: Union[str, Path]) -> DeadCodeAnalysisResult:
        """
        Analyze a directory for dead code issues.
        
        Args:
            directory: Path to the directory to analyze
            
        Returns:
            DeadCodeAnalysisResult containing all analysis results
        """
        start_time = time.time()
        
        self.project_root = Path(directory)
        self.logger.info(f"Starting dead code analysis of {self.project_root}")
        
        # Find all Python files
        self.python_files = find_python_files(self.project_root)
        self.logger.info(f"Found {len(self.python_files)} Python files")
        
        # Phase 1: Parse all files and build dependency graph
        self._parse_all_files()
        
        # Phase 2: Build usage maps and public API detection
        self._build_usage_maps()
        
        # Phase 3: Analyze each file for dead code
        self._analyze_dead_code()
        
        # Phase 4: Generate results
        execution_time = time.time() - start_time
        result = self._generate_results(execution_time)
        
        self.logger.info(f"Analysis complete. Found {len(self.issues)} issues in {execution_time:.2f}s")
        return result
    
    def _parse_all_files(self) -> None:
        """Parse all Python files and build AST trees."""
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                self.ast_trees[str(file_path)] = tree
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {file_path}: {e}")
    
    def _build_usage_maps(self) -> None:
        """Build usage maps and detect public APIs."""
        for file_path, tree in self.ast_trees.items():
            # Build import graph
            self._extract_imports(file_path, tree)
            
            # Detect public APIs
            self._detect_public_apis(file_path, tree)
            
            # Build usage map
            self._extract_usage(file_path, tree)
    
    def _extract_imports(self, file_path: str, tree: ast.AST) -> None:
        """Extract import statements and build import graph."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_graph[file_path].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.import_graph[file_path].add(node.module)
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        self.import_graph[file_path].add(full_name)
    
    def _detect_public_apis(self, file_path: str, tree: ast.AST) -> None:
        """Detect public APIs using __all__ declarations."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    self.public_apis[file_path].add(elt.value)
                                elif isinstance(elt, ast.Str):  # Python < 3.8
                                    self.public_apis[file_path].add(elt.s)
    
    def _extract_usage(self, file_path: str, tree: ast.AST) -> None:
        """Extract usage patterns from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.usage_map[file_path].add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle attribute access like obj.method
                if isinstance(node.value, ast.Name):
                    self.usage_map[file_path].add(f"{node.value.id}.{node.attr}")
    
    def _analyze_dead_code(self) -> None:
        """Analyze each file for dead code issues."""
        for file_path, tree in self.ast_trees.items():
            file_issues = []
            
            # Analyze different types of dead code
            file_issues.extend(self._check_unused_functions(file_path, tree))
            file_issues.extend(self._check_unused_classes(file_path, tree))
            file_issues.extend(self._check_unused_imports(file_path, tree))
            file_issues.extend(self._check_unused_import_from(file_path, tree))
            
            self.issues.extend(file_issues)
            
            # Store file analysis
            self.file_analysis[file_path] = {
                "total_issues": len(file_issues),
                "issues_by_type": Counter(issue.issue_type for issue in file_issues),
                "issues_by_severity": Counter(issue.severity for issue in file_issues),
                "issues_by_confidence": self._group_by_confidence(file_issues),
                "public_apis": list(self.public_apis[file_path]),
                "imports": list(self.import_graph[file_path]),
                "usage": list(self.usage_map[file_path])
            }
    
    def _check_unused_functions(self, file_path: str, tree: ast.AST) -> List[DeadCodeIssue]:
        """Check for unused functions."""
        issues = []
        defined_functions = set()
        called_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_functions.add(node.func.attr)
        
        for func_name in defined_functions:
            if func_name not in called_functions:
                # Check if it's a public API or used elsewhere
                is_public = self._is_public_api(file_path, func_name)
                is_used_elsewhere = self._is_used_in_project(func_name, file_path)
                
                if not is_public and not is_used_elsewhere:
                    confidence = 0.9
                    severity = "medium"
                elif is_public:
                    confidence = 0.1
                    severity = "low"
                else:
                    confidence = 0.3
                    severity = "low"
                
                issues.append(DeadCodeIssue(
                    file_path=file_path,
                    line_number=0,  # Will be filled by caller
                    issue_type="unused_function",
                    name=func_name,
                    description=f"Function '{func_name}' is defined but never called",
                    confidence=confidence,
                    severity=severity,
                    is_public_api=is_public,
                    is_used_cross_file=is_used_elsewhere
                ))
        
        return issues
    
    def _check_unused_classes(self, file_path: str, tree: ast.AST) -> List[DeadCodeIssue]:
        """Check for unused classes."""
        issues = []
        defined_classes = set()
        used_classes = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined_classes.add(node.name)
                # Check if it's abstract or interface
                is_abstract = self._is_abstract_or_interface(node)
            elif isinstance(node, ast.Name):
                if node.id in defined_classes:
                    used_classes.add(node.id)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in defined_classes:
                    used_classes.add(node.func.id)
        
        for class_name in defined_classes:
            if class_name not in used_classes:
                is_public = self._is_public_api(file_path, class_name)
                is_used_elsewhere = self._is_used_in_project(class_name, file_path)
                is_abstract = any(
                    self._is_abstract_or_interface(node) 
                    for node in ast.walk(tree) 
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                
                if not is_public and not is_used_elsewhere and not is_abstract:
                    confidence = 0.8
                    severity = "medium"
                elif is_public or is_abstract:
                    confidence = 0.1
                    severity = "low"
                else:
                    confidence = 0.3
                    severity = "low"
                
                issues.append(DeadCodeIssue(
                    file_path=file_path,
                    line_number=0,
                    issue_type="unused_class",
                    name=class_name,
                    description=f"Class '{class_name}' is defined but never used",
                    confidence=confidence,
                    severity=severity,
                    is_public_api=is_public,
                    is_used_cross_file=is_used_elsewhere,
                    is_abstract_interface=is_abstract
                ))
        
        return issues
    
    def _check_unused_imports(self, file_path: str, tree: ast.AST) -> List[DeadCodeIssue]:
        """Check for unused import statements."""
        issues = []
        imports = set()
        usage = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.Name):
                usage.add(node.id)
        
        for import_name in imports:
            if import_name not in usage:
                confidence = 0.95  # High confidence for unused imports
                severity = "low"
                
                issues.append(DeadCodeIssue(
                    file_path=file_path,
                    line_number=0,
                    issue_type="unused_import",
                    name=import_name,
                    description=f"Import '{import_name}' is not used",
                    confidence=confidence,
                    severity=severity
                ))
        
        return issues
    
    def _check_unused_import_from(self, file_path: str, tree: ast.AST) -> List[DeadCodeIssue]:
        """Check for unused import from statements."""
        issues = []
        imports = set()
        usage = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.Name):
                usage.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    usage.add(f"{node.value.id}.{node.attr}")
        
        for import_name in imports:
            if import_name not in usage:
                confidence = 0.95
                severity = "low"
                
                issues.append(DeadCodeIssue(
                    file_path=file_path,
                    line_number=0,
                    issue_type="unused_import_from",
                    name=import_name,
                    description=f"Import '{import_name}' from module is not used",
                    confidence=confidence,
                    severity=severity
                ))
        
        return issues
    
    def _is_public_api(self, file_path: str, name: str) -> bool:
        """Check if a name is exposed in __all__."""
        return name in self.public_apis[file_path]
    
    def _is_used_in_project(self, name: str, current_file: str) -> bool:
        """Check if a name is used in any other file in the project."""
        for file_path, usage_set in self.usage_map.items():
            if file_path != current_file and name in usage_set:
                return True
        return False
    
    def _is_abstract_or_interface(self, node: ast.ClassDef) -> bool:
        """Check if a class is abstract or interface."""
        # Check for abstract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                        return True
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                        return True
        
        # Check for ABC inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in ["ABC", "AbstractBaseClass", "Protocol"]:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in ["ABC", "AbstractBaseClass", "Protocol"]:
                    return True
        
        return False
    
    def _group_by_confidence(self, issues: List[DeadCodeIssue]) -> Dict[str, int]:
        """Group issues by confidence level."""
        groups = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            if issue.confidence >= 0.8:
                groups["high"] += 1
            elif issue.confidence >= 0.5:
                groups["medium"] += 1
            else:
                groups["low"] += 1
        return groups
    
    def _generate_results(self, execution_time: float) -> DeadCodeAnalysisResult:
        """Generate comprehensive analysis results."""
        # Calculate summary statistics
        issues_by_type = Counter(issue.issue_type for issue in self.issues)
        issues_by_severity = Counter(issue.severity for issue in self.issues)
        issues_by_confidence = self._group_by_confidence(self.issues)
        
        # Global analysis
        global_analysis = {
            "total_files": len(self.python_files),
            "total_issues": len(self.issues),
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "issues_by_confidence": issues_by_confidence,
            "high_confidence_issues": len([i for i in self.issues if i.confidence >= 0.8]),
            "public_api_issues": len([i for i in self.issues if i.is_public_api]),
            "cross_file_usage_issues": len([i for i in self.issues if i.is_used_cross_file]),
            "abstract_interface_issues": len([i for i in self.issues if i.is_abstract_interface])
        }
        
        # Summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "execution_time": execution_time,
            "files_analyzed": len(self.python_files),
            "total_issues": len(self.issues),
            "issues_by_type": dict(issues_by_type),
            "issues_by_confidence": issues_by_confidence,
            "issues_by_severity": dict(issues_by_severity),
            "recommendations": self._generate_recommendations()
        }
        
        return DeadCodeAnalysisResult(
            issues=self.issues,
            summary=summary,
            file_analysis=self.file_analysis,
            global_analysis=global_analysis,
            execution_time=execution_time,
            files_analyzed=len(self.python_files),
            total_issues=len(self.issues),
            issues_by_type=dict(issues_by_type),
            issues_by_confidence=issues_by_confidence,
            issues_by_severity=dict(issues_by_severity)
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        high_conf_issues = [i for i in self.issues if i.confidence >= 0.8]
        if high_conf_issues:
            recommendations.append(f"Consider removing {len(high_conf_issues)} high-confidence unused imports")
        
        unused_functions = [i for i in self.issues if i.issue_type == "unused_function" and i.confidence >= 0.8]
        if unused_functions:
            recommendations.append(f"Review {len(unused_functions)} potentially unused functions")
        
        unused_classes = [i for i in self.issues if i.issue_type == "unused_class" and i.confidence >= 0.8]
        if unused_classes:
            recommendations.append(f"Review {len(unused_classes)} potentially unused classes")
        
        public_api_issues = [i for i in self.issues if i.is_public_api]
        if public_api_issues:
            recommendations.append(f"Verify {len(public_api_issues)} public API items are intentionally exposed")
        
        return recommendations
    
    def save_report(self, output_path: Union[str, Path]) -> Path:
        """Save analysis results to a JSON file."""
        output_path = Path(output_path)
        
        # Convert dataclasses to dictionaries for JSON serialization
        report_data = {
            "summary": self._generate_results(0).summary,
            "global_analysis": self._generate_results(0).global_analysis,
            "file_analysis": self.file_analysis,
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "issue_type": issue.issue_type,
                    "name": issue.name,
                    "description": issue.description,
                    "confidence": issue.confidence,
                    "severity": issue.severity,
                    "is_public_api": issue.is_public_api,
                    "is_used_cross_file": issue.is_used_cross_file,
                    "is_abstract_interface": issue.is_abstract_interface,
                    "removal_impact": issue.removal_impact,
                    "dependencies": issue.dependencies
                }
                for issue in self.issues
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Report saved to {output_path}")
        return output_path


def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Dead Code Analyzer")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    analyzer = ImprovedDeadCodeAnalyzer()
    result = analyzer.analyze_directory(args.directory)
    
    # Save report
    if args.output:
        analyzer.save_report(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.directory) / f"improved_dead_code_analysis_{timestamp}.json"
        analyzer.save_report(output_path)
    
    # Print summary
    print(f"\nDead Code Analysis Summary:")
    print(f"Files analyzed: {result.files_analyzed}")
    print(f"Total issues: {result.total_issues}")
    print(f"High confidence issues: {result.global_analysis['high_confidence_issues']}")
    print(f"Execution time: {result.execution_time:.2f}s")


if __name__ == "__main__":
    main()
