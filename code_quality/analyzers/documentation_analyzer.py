#!/usr/bin/env python3
"""
Documentation Quality Analyzer

Analyzes documentation quality including:
- Docstring completeness and quality
- Parameter documentation
- Return value documentation
- Example usage presence
- Comment quality and relevance
- README quality assessment
- API documentation coverage
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import docstring_parser


@dataclass 
class DocstringIssue:
    """Represents a documentation issue."""
    file_path: str
    line_number: int
    entity_name: str
    entity_type: str  # 'module', 'class', 'function', 'method'
    issue_type: str
    message: str
    severity: str  # 'error', 'warning', 'info'


@dataclass
class DocstringMetrics:
    """Metrics for docstring quality."""
    has_docstring: bool
    has_description: bool
    has_parameters: bool
    has_returns: bool
    has_raises: bool
    has_examples: bool
    completeness_score: float
    quality_score: float
    word_count: int
    
    
@dataclass
class CommentMetrics:
    """Metrics for code comments."""
    total_comments: int
    inline_comments: int
    block_comments: int
    todo_comments: int
    fixme_comments: int
    comment_to_code_ratio: float
    avg_comment_length: float


class DocumentationAnalyzer:
    """Analyzes documentation quality in Python projects."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[DocstringIssue] = []
        self.file_metrics: Dict[str, Dict[str, Any]] = {}
        self.readme_quality: Dict[str, Any] = {}
        
        # Docstring style patterns
        self.docstring_styles = {
            'google': {
                'params': r'Args?:\s*\n',
                'returns': r'Returns?:\s*\n',
                'raises': r'Raises?:\s*\n',
                'examples': r'Examples?:\s*\n'
            },
            'numpy': {
                'params': r'Parameters\s*\n\s*-+\s*\n',
                'returns': r'Returns\s*\n\s*-+\s*\n', 
                'raises': r'Raises\s*\n\s*-+\s*\n',
                'examples': r'Examples\s*\n\s*-+\s*\n'
            },
            'sphinx': {
                'params': r':param\s+\w+:',
                'returns': r':returns?:',
                'raises': r':raises?\s+\w+:',
                'examples': r'\.\..*code-block::'
            }
        }
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze documentation in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            tree = ast.parse(content, filename=str(file_path))
            
            # Analyze docstrings
            analyzer = DocstringAnalyzer(str(file_path), content, lines, self)
            analyzer.visit(tree)
            
            # Analyze comments
            comment_metrics = self._analyze_comments(lines)
            
            # Calculate file-level metrics
            file_metrics = {
                'docstring_coverage': analyzer.get_coverage(),
                'docstring_quality': analyzer.get_average_quality(),
                'comment_metrics': comment_metrics,
                'issues': analyzer.issues
            }
            
            self.file_metrics[str(file_path)] = file_metrics
            return file_metrics
            
        except Exception as e:
            return {'error': str(e)}
            
    def analyze_readme(self, readme_path: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze README quality."""
        if not readme_path:
            # Look for README files
            for pattern in ['README.md', 'README.rst', 'README.txt', 'readme.md']:
                readme_path = self.project_root / pattern
                if readme_path.exists():
                    break
            else:
                return {'found': False, 'message': 'No README file found'}
                
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for essential sections
            sections = {
                'description': bool(re.search(r'^#[^#].*', content, re.MULTILINE)),
                'installation': bool(re.search(r'(?i)(installation|install|setup)', content)),
                'usage': bool(re.search(r'(?i)(usage|example|how to)', content)),
                'requirements': bool(re.search(r'(?i)(requirements|dependencies)', content)),
                'license': bool(re.search(r'(?i)license', content)),
                'contributing': bool(re.search(r'(?i)contribut', content)),
                'api_docs': bool(re.search(r'(?i)(api|reference|documentation)', content))
            }
            
            # Calculate quality metrics
            word_count = len(content.split())
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            has_badges = bool(re.search(r'!\[.*\]\(.*\)', content))
            has_toc = bool(re.search(r'(?i)table of contents|## contents', content))
            
            completeness = sum(sections.values()) / len(sections) * 100
            
            self.readme_quality = {
                'found': True,
                'path': str(readme_path),
                'sections': sections,
                'word_count': word_count,
                'code_blocks': code_blocks,
                'has_badges': has_badges,
                'has_toc': has_toc,
                'completeness_score': completeness,
                'missing_sections': [k for k, v in sections.items() if not v]
            }
            
            return self.readme_quality
            
        except Exception as e:
            return {'found': True, 'error': str(e)}
            
    def _analyze_comments(self, lines: List[str]) -> CommentMetrics:
        """Analyze comment quality in code."""
        total_comments = 0
        inline_comments = 0
        block_comments = 0
        todo_comments = 0
        fixme_comments = 0
        comment_lengths = []
        code_lines = 0
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Track docstrings to not count them as comments
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    in_docstring = True
                    docstring_delimiter = '"""' if '"""' in line else "'''"
                elif docstring_delimiter in line:
                    in_docstring = False
                    docstring_delimiter = None
                continue
                
            if in_docstring:
                continue
                
            # Count actual code lines
            if stripped and not stripped.startswith('#'):
                code_lines += 1
                
            # Analyze comments
            if '#' in line and not in_docstring:
                comment_start = line.find('#')
                comment_text = line[comment_start+1:].strip()
                
                if comment_text:
                    total_comments += 1
                    comment_lengths.append(len(comment_text))
                    
                    # Check comment position
                    if line[:comment_start].strip():  # Code before comment
                        inline_comments += 1
                    else:
                        block_comments += 1
                        
                    # Check for TODOs and FIXMEs
                    if 'TODO' in comment_text.upper():
                        todo_comments += 1
                    if 'FIXME' in comment_text.upper():
                        fixme_comments += 1
                        
        comment_ratio = total_comments / code_lines if code_lines > 0 else 0
        avg_length = sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0
        
        return CommentMetrics(
            total_comments=total_comments,
            inline_comments=inline_comments,
            block_comments=block_comments,
            todo_comments=todo_comments,
            fixme_comments=fixme_comments,
            comment_to_code_ratio=comment_ratio,
            avg_comment_length=avg_length
        )
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive documentation quality report."""
        total_files = len(self.file_metrics)
        
        # Aggregate metrics
        total_entities = 0
        documented_entities = 0
        total_issues = len(self.issues)
        
        for file_metrics in self.file_metrics.values():
            coverage = file_metrics.get('docstring_coverage', {})
            total_entities += coverage.get('total_entities', 0)
            documented_entities += coverage.get('documented_entities', 0)
            
        overall_coverage = (documented_entities / total_entities * 100) if total_entities > 0 else 0
        
        # Find files with poor documentation
        poor_docs_files = []
        for file_path, metrics in self.file_metrics.items():
            coverage = metrics.get('docstring_coverage', {})
            if coverage.get('coverage_percentage', 100) < 50:
                poor_docs_files.append({
                    'file': file_path,
                    'coverage': coverage.get('coverage_percentage', 0),
                    'missing': coverage.get('missing_docstrings', [])
                })
                
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.issue_type].append({
                'file': issue.file_path,
                'line': issue.line_number,
                'entity': issue.entity_name,
                'message': issue.message
            })
            
        return {
            'summary': {
                'total_files': total_files,
                'total_entities': total_entities,
                'documented_entities': documented_entities,
                'overall_coverage': overall_coverage,
                'total_issues': total_issues,
                'readme_quality': self.readme_quality.get('completeness_score', 0)
            },
            'poor_documentation_files': sorted(poor_docs_files, key=lambda x: x['coverage']),
            'issues_by_type': dict(issues_by_type),
            'readme_analysis': self.readme_quality,
            'file_details': {
                file: {
                    'coverage': metrics['docstring_coverage'],
                    'quality': metrics['docstring_quality'],
                    'comments': {
                        'total': metrics['comment_metrics'].total_comments,
                        'ratio': metrics['comment_metrics'].comment_to_code_ratio,
                        'todos': metrics['comment_metrics'].todo_comments,
                        'fixmes': metrics['comment_metrics'].fixme_comments
                    }
                }
                for file, metrics in self.file_metrics.items()
            }
        }


class DocstringAnalyzer(ast.NodeVisitor):
    """Analyzes docstrings in Python code."""
    
    def __init__(self, file_path: str, content: str, lines: List[str], analyzer: DocumentationAnalyzer):
        self.file_path = file_path
        self.content = content
        self.lines = lines
        self.analyzer = analyzer
        self.issues = []
        
        # Tracking
        self.total_entities = 0
        self.documented_entities = 0
        self.docstring_metrics = []
        self.current_class = None
        
    def visit_Module(self, node: ast.Module) -> None:
        """Check module-level docstring."""
        self.total_entities += 1
        docstring = ast.get_docstring(node)
        
        if docstring:
            self.documented_entities += 1
        else:
            self._add_issue(
                1, 'module', 'module',
                'missing_docstring',
                'Module lacks a docstring',
                'warning'
            )
            
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class docstring."""
        self.total_entities += 1
        docstring = ast.get_docstring(node)
        
        if docstring:
            self.documented_entities += 1
            metrics = self._analyze_docstring(docstring, node, 'class')
            self.docstring_metrics.append(metrics)
        else:
            self._add_issue(
                node.lineno, node.name, 'class',
                'missing_docstring',
                'Class lacks a docstring',
                'error'
            )
            
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function/method docstring."""
        self._check_function_docs(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function/method docstring."""
        self._check_function_docs(node)
        
    def _check_function_docs(self, node: Any) -> None:
        """Check function documentation."""
        # Skip private methods and special methods from coverage requirements
        if node.name.startswith('_') and not node.name.startswith('__'):
            return
            
        self.total_entities += 1
        docstring = ast.get_docstring(node)
        entity_type = 'method' if self.current_class else 'function'
        
        if docstring:
            self.documented_entities += 1
            metrics = self._analyze_docstring(docstring, node, entity_type)
            self.docstring_metrics.append(metrics)
            
            # Check for specific documentation requirements
            if len(node.args.args) > 1:  # Has parameters (excluding self)
                if not metrics.has_parameters:
                    self._add_issue(
                        node.lineno, node.name, entity_type,
                        'missing_parameter_docs',
                        'Function has parameters but no parameter documentation',
                        'warning'
                    )
                    
            # Check for return documentation
            if self._has_return_value(node) and not metrics.has_returns:
                self._add_issue(
                    node.lineno, node.name, entity_type,
                    'missing_return_docs',
                    'Function returns a value but lacks return documentation',
                    'warning'
                )
        else:
            self._add_issue(
                node.lineno, node.name, entity_type,
                'missing_docstring',
                f'{entity_type.capitalize()} lacks a docstring',
                'error' if not node.name.startswith('_') else 'warning'
            )
            
        self.generic_visit(node)
        
    def _analyze_docstring(self, docstring: str, node: ast.AST, entity_type: str) -> DocstringMetrics:
        """Analyze docstring quality and completeness."""
        # Basic metrics
        has_description = len(docstring.strip()) > 0
        word_count = len(docstring.split())
        
        # Detect style and check sections
        style = self._detect_docstring_style(docstring)
        has_params = False
        has_returns = False  
        has_raises = False
        has_examples = False
        
        if style:
            patterns = self.analyzer.docstring_styles[style]
            has_params = bool(re.search(patterns['params'], docstring))
            has_returns = bool(re.search(patterns['returns'], docstring))
            has_raises = bool(re.search(patterns['raises'], docstring))
            has_examples = bool(re.search(patterns['examples'], docstring))
            
        # Calculate scores
        sections = [has_description, has_params, has_returns, has_raises, has_examples]
        completeness = sum(sections) / len(sections)
        
        # Quality score based on length and completeness
        quality = min(1.0, (word_count / 50) * 0.5 + completeness * 0.5)
        
        return DocstringMetrics(
            has_docstring=True,
            has_description=has_description,
            has_parameters=has_params,
            has_returns=has_returns,
            has_raises=has_raises,
            has_examples=has_examples,
            completeness_score=completeness,
            quality_score=quality,
            word_count=word_count
        )
        
    def _detect_docstring_style(self, docstring: str) -> Optional[str]:
        """Detect the docstring style being used."""
        for style, patterns in self.analyzer.docstring_styles.items():
            if any(re.search(pattern, docstring) for pattern in patterns.values()):
                return style
        return None
        
    def _has_return_value(self, node: ast.FunctionDef) -> bool:
        """Check if function has a return value."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False
        
    def _add_issue(self, line_number: int, entity_name: str, entity_type: str,
                   issue_type: str, message: str, severity: str) -> None:
        """Add a documentation issue."""
        issue = DocstringIssue(
            file_path=self.file_path,
            line_number=line_number,
            entity_name=entity_name,
            entity_type=entity_type,
            issue_type=issue_type,
            message=message,
            severity=severity
        )
        self.issues.append(issue)
        self.analyzer.issues.append(issue)
        
    def get_coverage(self) -> Dict[str, Any]:
        """Get docstring coverage statistics."""
        return {
            'total_entities': self.total_entities,
            'documented_entities': self.documented_entities,
            'coverage_percentage': (self.documented_entities / self.total_entities * 100) if self.total_entities > 0 else 100,
            'missing_docstrings': [
                issue.entity_name for issue in self.issues
                if issue.issue_type == 'missing_docstring'
            ]
        }
        
    def get_average_quality(self) -> float:
        """Get average docstring quality score."""
        if not self.docstring_metrics:
            return 0.0
        return sum(m.quality_score for m in self.docstring_metrics) / len(self.docstring_metrics)