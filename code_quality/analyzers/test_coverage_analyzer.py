#!/usr/bin/env python3
"""
Test Coverage Analyzer

Analyzes test coverage and quality including:
- Test file detection
- Test-to-code ratio
- Missing test detection
- Test quality metrics
- Assertion density
- Mock usage analysis
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TestFileMetrics:
    """Metrics for a test file."""
    file_path: str
    test_count: int
    assertion_count: int
    mock_count: int
    fixture_count: int
    test_lines: int
    setup_teardown_count: int
    parametrized_tests: int
    skipped_tests: int


@dataclass
class TestCoverage:
    """Test coverage information for a source file."""
    source_file: str
    test_files: List[str]
    tested_functions: Set[str]
    untested_functions: Set[str]
    test_count: int
    coverage_percentage: float


@dataclass
class TestQualityIssue:
    """Represents a test quality issue."""
    file_path: str
    line_number: int
    issue_type: str
    message: str
    severity: str  # 'error', 'warning', 'info'


class TestCoverageAnalyzer:
    """Analyzes test coverage and quality."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_files: Dict[str, TestFileMetrics] = {}
        self.source_files: Dict[str, Set[str]] = {}  # source file -> functions
        self.test_coverage: Dict[str, TestCoverage] = {}
        self.quality_issues: List[TestQualityIssue] = []
        
        # Test frameworks patterns
        self.test_patterns = {
            'pytest': {
                'test_function': re.compile(r'^test_\w+'),
                'test_class': re.compile(r'^Test\w+'),
                'assertion': ['assert ', 'pytest.raises', 'pytest.warns'],
                'fixture': '@pytest.fixture',
                'parametrize': '@pytest.mark.parametrize',
                'skip': '@pytest.mark.skip'
            },
            'unittest': {
                'test_function': re.compile(r'^test_\w+'),
                'test_class': re.compile(r'^Test\w+'),
                'assertion': ['self.assert', 'self.fail'],
                'fixture': 'setUp',
                'parametrize': None,
                'skip': '@unittest.skip'
            }
        }
        
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the entire project for test coverage."""
        # Find all Python files
        all_files = list(self.project_root.rglob('*.py'))
        
        # Separate test and source files
        test_files = []
        source_files = []
        
        for file in all_files:
            if self._is_test_file(file):
                test_files.append(file)
            else:
                source_files.append(file)
                
        # Analyze source files
        for source_file in source_files:
            self._analyze_source_file(source_file)
            
        # Analyze test files
        for test_file in test_files:
            self._analyze_test_file(test_file)
            
        # Calculate coverage
        self._calculate_coverage()
        
        # Generate report
        return self._generate_report()
        
    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file."""
        path_str = str(file_path)
        file_name = file_path.name
        
        # Common test file patterns
        test_indicators = [
            'test_' in file_name,
            '_test.py' in file_name,
            '/tests/' in path_str,
            '/test/' in path_str,
            'tests.py' == file_name
        ]
        
        return any(test_indicators)
        
    def _analyze_source_file(self, file_path: Path) -> None:
        """Analyze a source file to extract function definitions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract all function names
            functions = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip private methods and special methods for coverage
                    if not node.name.startswith('_'):
                        functions.add(node.name)
                        
            self.source_files[str(file_path)] = functions
            
        except Exception as e:
            pass  # Skip files that can't be parsed
            
    def _analyze_test_file(self, file_path: Path) -> None:
        """Analyze a test file for metrics and coverage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            tree = ast.parse(content, filename=str(file_path))
            
            # Analyze the test file
            analyzer = TestFileAnalyzer(content, lines, self)
            analyzer.visit(tree)
            
            # Store metrics
            self.test_files[str(file_path)] = analyzer.get_metrics()
            
        except Exception as e:
            self._add_issue(
                str(file_path), 0, 'parse_error',
                f'Failed to parse test file: {str(e)}', 'error'
            )
            
    def _calculate_coverage(self) -> None:
        """Calculate test coverage for each source file."""
        # Map test files to source files they test
        test_to_source_mapping = self._map_tests_to_sources()
        
        for source_file, functions in self.source_files.items():
            tested_functions = set()
            test_files = []
            test_count = 0
            
            # Find tests for this source file
            source_name = Path(source_file).stem
            
            for test_file, tested in test_to_source_mapping.items():
                if source_name in tested or source_file in tested:
                    test_files.append(test_file)
                    test_count += self.test_files.get(test_file, TestFileMetrics(test_file, 0, 0, 0, 0, 0, 0, 0, 0)).test_count
                    
                    # Assume functions with matching names are tested
                    for func in functions:
                        if func in tested or f'test_{func}' in tested:
                            tested_functions.add(func)
                            
            untested_functions = functions - tested_functions
            coverage_percentage = (len(tested_functions) / len(functions) * 100) if functions else 100
            
            self.test_coverage[source_file] = TestCoverage(
                source_file=source_file,
                test_files=test_files,
                tested_functions=tested_functions,
                untested_functions=untested_functions,
                test_count=test_count,
                coverage_percentage=coverage_percentage
            )
            
    def _map_tests_to_sources(self) -> Dict[str, Set[str]]:
        """Map test files to the sources they test."""
        mapping = {}
        
        for test_file in self.test_files:
            # Extract what this test file tests based on imports and test names
            tested = set()
            
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for imports
                import_pattern = re.compile(r'from\s+(\S+)\s+import|import\s+(\S+)')
                for match in import_pattern.finditer(content):
                    module = match.group(1) or match.group(2)
                    if not module.startswith('.'):
                        tested.add(module)
                        
                # Look for test function names that might indicate what they test
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith('test_'):
                            # Extract what might be tested from the name
                            tested_item = node.name[5:]  # Remove 'test_'
                            tested.add(tested_item)
                            
            except:
                pass
                
            mapping[test_file] = tested
            
        return mapping
        
    def _add_issue(self, file_path: str, line_number: int, issue_type: str,
                   message: str, severity: str) -> None:
        """Add a test quality issue."""
        self.quality_issues.append(TestQualityIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            message=message,
            severity=severity
        ))
        
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test coverage report."""
        total_source_files = len(self.source_files)
        total_test_files = len(self.test_files)
        total_functions = sum(len(funcs) for funcs in self.source_files.values())
        tested_functions = sum(len(cov.tested_functions) for cov in self.test_coverage.values())
        
        # Calculate metrics
        total_tests = sum(m.test_count for m in self.test_files.values())
        total_assertions = sum(m.assertion_count for m in self.test_files.values())
        assertion_density = total_assertions / total_tests if total_tests > 0 else 0
        
        # Find untested files
        untested_files = [
            file for file, cov in self.test_coverage.items()
            if cov.coverage_percentage == 0
        ]
        
        # Find files with low coverage
        low_coverage_files = [
            {
                'file': file,
                'coverage': cov.coverage_percentage,
                'untested_functions': list(cov.untested_functions)
            }
            for file, cov in self.test_coverage.items()
            if 0 < cov.coverage_percentage < 80
        ]
        
        return {
            'summary': {
                'total_source_files': total_source_files,
                'total_test_files': total_test_files,
                'total_functions': total_functions,
                'tested_functions': tested_functions,
                'overall_coverage': (tested_functions / total_functions * 100) if total_functions > 0 else 0,
                'total_tests': total_tests,
                'total_assertions': total_assertions,
                'assertion_density': assertion_density,
                'test_to_code_ratio': total_test_files / total_source_files if total_source_files > 0 else 0
            },
            'untested_files': untested_files,
            'low_coverage_files': sorted(low_coverage_files, key=lambda x: x['coverage']),
            'test_quality_issues': [
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.issue_type,
                    'message': issue.message,
                    'severity': issue.severity
                }
                for issue in self.quality_issues
            ],
            'test_metrics': {
                file: {
                    'tests': metrics.test_count,
                    'assertions': metrics.assertion_count,
                    'mocks': metrics.mock_count,
                    'fixtures': metrics.fixture_count,
                    'parametrized': metrics.parametrized_tests,
                    'skipped': metrics.skipped_tests
                }
                for file, metrics in self.test_files.items()
            }
        }


class TestFileAnalyzer(ast.NodeVisitor):
    """Analyzes a test file for metrics."""
    
    def __init__(self, content: str, lines: List[str], analyzer: TestCoverageAnalyzer):
        self.content = content
        self.lines = lines
        self.analyzer = analyzer
        self.test_count = 0
        self.assertion_count = 0
        self.mock_count = 0
        self.fixture_count = 0
        self.setup_teardown_count = 0
        self.parametrized_tests = 0
        self.skipped_tests = 0
        self.current_test = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        # Check if it's a test function
        if node.name.startswith('test_'):
            self.test_count += 1
            self.current_test = node.name
            
            # Check for test without assertions
            has_assertion = False
            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    has_assertion = True
                    break
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if 'assert' in child.func.attr:
                        has_assertion = True
                        break
                        
            if not has_assertion:
                self.analyzer._add_issue(
                    self.analyzer.current_file, node.lineno,
                    'missing_assertion',
                    f"Test '{node.name}' has no assertions",
                    'warning'
                )
                
            # Check for parametrized tests
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and 'parametrize' in ast.unparse(decorator):
                    self.parametrized_tests += 1
                elif isinstance(decorator, ast.Attribute) and 'skip' in ast.unparse(decorator):
                    self.skipped_tests += 1
                    
        elif node.name in ['setUp', 'tearDown', 'setup_method', 'teardown_method']:
            self.setup_teardown_count += 1
            
        # Check for fixtures
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute) and 'fixture' in ast.unparse(decorator):
                self.fixture_count += 1
                
        self.generic_visit(node)
        self.current_test = None
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)  # Treat the same as regular functions
        
    def visit_Assert(self, node: ast.Assert) -> None:
        """Visit assert statement."""
        self.assertion_count += 1
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        # Count assertion methods
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if 'assert' in attr_name.lower() or attr_name in ['fail', 'raises', 'warns']:
                self.assertion_count += 1
                
        # Count mock usage
        if isinstance(node.func, ast.Name):
            if 'mock' in node.func.id.lower() or node.func.id in ['Mock', 'MagicMock', 'patch']:
                self.mock_count += 1
        elif isinstance(node.func, ast.Attribute):
            if 'mock' in ast.unparse(node.func).lower():
                self.mock_count += 1
                
        self.generic_visit(node)
        
    def get_metrics(self) -> TestFileMetrics:
        """Get the collected metrics."""
        return TestFileMetrics(
            file_path=self.analyzer.current_file,
            test_count=self.test_count,
            assertion_count=self.assertion_count,
            mock_count=self.mock_count,
            fixture_count=self.fixture_count,
            test_lines=len(self.lines),
            setup_teardown_count=self.setup_teardown_count,
            parametrized_tests=self.parametrized_tests,
            skipped_tests=self.skipped_tests
        )