"""
Code Complexity Analyzer

Analyzes Python code complexity using Radon library.
Provides cyclomatic complexity, maintainability index, and Halstead metrics.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import radon.complexity as radon_complexity
import radon.metrics as radon_metrics
from radon.visitors import ComplexityVisitor, HalsteadVisitor

from ..core.config import AnalysisConfig
from minimal_file_utils import find_python_files


@dataclass
class ComplexityMetrics:
    """Container for complexity analysis results."""
    cyclomatic_complexity: int
    maintainability_index: float
    halstead_volume: float
    halstead_difficulty: float
    halstead_effort: float
    halstead_time: float
    halstead_bugs: float
    loc: int
    lloc: int
    sloc: int
    comments: int
    multi: int
    blank: int


@dataclass
class FunctionComplexity:
    """Container for function-level complexity analysis."""
    name: str
    lineno: int
    endline: int
    complexity: int
    signature: str
    metrics: ComplexityMetrics


@dataclass
class ClassComplexity:
    """Container for class-level complexity analysis."""
    name: str
    lineno: int
    endline: int
    complexity: int
    methods: List[FunctionComplexity]
    metrics: ComplexityMetrics


@dataclass
class ModuleComplexity:
    """Container for module-level complexity analysis."""
    path: str
    functions: List[FunctionComplexity]
    classes: List[ClassComplexity]
    overall_metrics: ComplexityMetrics
    complexity_score: float


class ComplexityAnalyzer:
    """
    Analyzes code complexity using Radon library.
    
    Provides comprehensive complexity analysis including:
    - Cyclomatic complexity
    - Maintainability index
    - Halstead metrics
    - Code structure analysis
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the complexity analyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()
        self.complexity_threshold = getattr(self.config, 'complexity_threshold', 10)
        self.maintainability_threshold = getattr(self.config, 'maintainability_threshold', 65)
        
    def analyze_file(self, file_path: Union[str, Path]) -> ModuleComplexity:
        """
        Analyze complexity of a single Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            ModuleComplexity object with analysis results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.suffix == '.py':
            raise ValueError(f"File must be a Python file: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        return self._analyze_source(source, str(file_path))
    
    def analyze_directory(self, directory: Union[str, Path]) -> Dict[str, ModuleComplexity]:
        """
        Analyze complexity of all Python files in a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Dictionary mapping file paths to ModuleComplexity objects
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
            
        python_files = find_python_files(directory)
        results = {}
        
        for file_path in python_files:
            try:
                results[str(file_path)] = self.analyze_file(file_path)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                
        return results
    
    def analyze_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, ModuleComplexity]:
        """
        Analyze complexity of multiple Python files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to ModuleComplexity objects
        """
        results = {}
        
        for file_path in file_paths:
            try:
                results[str(file_path)] = self.analyze_file(file_path)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                
        return results
    
    def _analyze_source(self, source: str, file_path: str) -> ModuleComplexity:
        """
        Analyze source code complexity.
        
        Args:
            source: Python source code
            file_path: File path for reference
            
        Returns:
            ModuleComplexity object
        """
        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        # Get basic metrics
        metrics = self._get_basic_metrics(source)
        
        # Analyze functions
        functions = self._analyze_functions(source, file_path)
        
        # Analyze classes
        classes = self._analyze_classes(source, file_path)
        
        # Calculate overall complexity score
        complexity_score = self._calculate_complexity_score(functions, classes, metrics)
        
        return ModuleComplexity(
            path=file_path,
            functions=functions,
            classes=classes,
            overall_metrics=metrics,
            complexity_score=complexity_score
        )
    
    def _get_basic_metrics(self, source: str) -> ComplexityMetrics:
        """Extract basic code metrics using Radon."""
        try:
            # Get raw metrics
            raw_metrics = radon_metrics.raw_metrics(source)
            
            # Get Halstead metrics
            halstead = HalsteadVisitor.from_ast(ast.parse(source))
            
            return ComplexityMetrics(
                cyclomatic_complexity=0,  # Will be calculated per function
                maintainability_index=0,  # Will be calculated overall
                halstead_volume=halstead.volume,
                halstead_difficulty=halstead.difficulty,
                halstead_effort=halstead.effort,
                halstead_time=halstead.time,
                halstead_bugs=halstead.bugs,
                loc=raw_metrics['loc'],
                lloc=raw_metrics['lloc'],
                sloc=raw_metrics['sloc'],
                comments=raw_metrics['comments'],
                multi=raw_metrics['multi'],
                blank=raw_metrics['blank']
            )
        except Exception:
            # Fallback to basic metrics
            lines = source.split('\n')
            return ComplexityMetrics(
                cyclomatic_complexity=0,
                maintainability_index=0,
                halstead_volume=0,
                halstead_difficulty=0,
                halstead_effort=0,
                halstead_time=0,
                halstead_bugs=0,
                loc=len(lines),
                lloc=len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
                sloc=len([l for l in lines if l.strip()]),
                comments=len([l for l in lines if l.strip().startswith('#')]),
                multi=0,
                blank=len([l for l in lines if not l.strip()])
            )
    
    def _analyze_functions(self, source: str, file_path: str) -> List[FunctionComplexity]:
        """Analyze function complexity."""
        functions = []
        
        try:
            # Use Radon complexity visitor
            visitor = ComplexityVisitor.from_ast(ast.parse(source))
            
            for func in visitor.functions:
                # Get function signature
                signature = self._extract_function_signature(source, func.lineno)
                
                # Calculate metrics for this function
                func_metrics = self._calculate_function_metrics(source, func.lineno, func.endline)
                
                functions.append(FunctionComplexity(
                    name=func.name,
                    lineno=func.lineno,
                    endline=func.endline,
                    complexity=func.complexity,
                    signature=signature,
                    metrics=func_metrics
                ))
        except Exception as e:
            print(f"Warning: Could not analyze functions in {file_path}: {e}")
            
        return functions
    
    def _analyze_classes(self, source: str, file_path: str) -> List[ClassComplexity]:
        """Analyze class complexity."""
        classes = []
        
        try:
            # Use Radon complexity visitor
            visitor = ComplexityVisitor.from_ast(ast.parse(source))
            
            for cls in visitor.classes:
                # Get class methods
                methods = []
                for method in cls.methods:
                    method_metrics = self._calculate_function_metrics(
                        source, method.lineno, method.endline
                    )
                    methods.append(FunctionComplexity(
                        name=method.name,
                        lineno=method.lineno,
                        endline=method.endline,
                        complexity=method.complexity,
                        signature=self._extract_function_signature(source, method.lineno),
                        metrics=method_metrics
                    ))
                
                # Calculate class metrics
                class_metrics = self._calculate_class_metrics(source, cls.lineno, cls.endline)
                
                classes.append(ClassComplexity(
                    name=cls.name,
                    lineno=cls.lineno,
                    endline=cls.endline,
                    complexity=cls.complexity,
                    methods=methods,
                    metrics=class_metrics
                ))
        except Exception as e:
            print(f"Warning: Could not analyze classes in {file_path}: {e}")
            
        return classes
    
    def _extract_function_signature(self, source: str, lineno: int) -> str:
        """Extract function signature from source."""
        lines = source.split('\n')
        if lineno <= len(lines):
            line = lines[lineno - 1].strip()
            # Find the opening parenthesis
            if '(' in line:
                return line[:line.find('(') + 1] + "...)"
            return line
        return "unknown"
    
    def _calculate_function_metrics(self, source: str, start_line: int, end_line: int) -> ComplexityMetrics:
        """Calculate metrics for a specific function."""
        lines = source.split('\n')[start_line-1:end_line]
        func_source = '\n'.join(lines)
        
        try:
            raw_metrics = radon_metrics.raw_metrics(func_source)
            halstead = HalsteadVisitor.from_ast(ast.parse(func_source))
            
            return ComplexityMetrics(
                cyclomatic_complexity=0,  # Already calculated by visitor
                maintainability_index=0,  # Will be calculated overall
                halstead_volume=halstead.volume,
                halstead_difficulty=halstead.difficulty,
                halstead_effort=halstead.effort,
                halstead_time=halstead.time,
                halstead_bugs=halstead.bugs,
                loc=raw_metrics['loc'],
                lloc=raw_metrics['lloc'],
                sloc=raw_metrics['sloc'],
                comments=raw_metrics['comments'],
                multi=raw_metrics['multi'],
                blank=raw_metrics['blank']
            )
        except Exception:
            return ComplexityMetrics(
                cyclomatic_complexity=0,
                maintainability_index=0,
                halstead_volume=0,
                halstead_difficulty=0,
                halstead_effort=0,
                halstead_time=0,
                halstead_bugs=0,
                loc=end_line - start_line + 1,
                lloc=end_line - start_line + 1,
                sloc=end_line - start_line + 1,
                comments=0,
                multi=0,
                blank=0
            )
    
    def _calculate_class_metrics(self, source: str, start_line: int, end_line: int) -> ComplexityMetrics:
        """Calculate metrics for a specific class."""
        return self._calculate_function_metrics(source, start_line, end_line)
    
    def _calculate_complexity_score(self, functions: List[FunctionComplexity], 
                                  classes: List[ClassComplexity], 
                                  metrics: ComplexityMetrics) -> float:
        """Calculate overall complexity score."""
        if not functions and not classes:
            return 0.0
            
        # Calculate maintainability index
        total_complexity = sum(f.complexity for f in functions) + sum(c.complexity for c in classes)
        total_loc = metrics.sloc
        
        if total_loc == 0:
            maintainability_index = 100.0
        else:
            # Simplified maintainability index calculation
            maintainability_index = max(0, 100 - (total_complexity * 2) - (total_loc / 10))
        
        # Update metrics
        metrics.maintainability_index = maintainability_index
        metrics.cyclomatic_complexity = total_complexity
        
        # Calculate overall score (0-100, higher is better)
        complexity_penalty = min(50, total_complexity * 2)
        maintainability_bonus = max(0, maintainability_index - 50) / 2
        
        return max(0, 100 - complexity_penalty + maintainability_bonus)
    
    def get_complexity_summary(self, results: Dict[str, ModuleComplexity]) -> Dict:
        """Generate complexity summary across multiple files."""
        summary = {
            'total_files': len(results),
            'total_functions': 0,
            'total_classes': 0,
            'high_complexity_functions': 0,
            'high_complexity_classes': 0,
            'low_maintainability_files': 0,
            'average_complexity_score': 0.0,
            'complexity_distribution': {
                'low': 0,      # 0-5
                'medium': 0,   # 6-10
                'high': 0,     # 11-20
                'very_high': 0 # 20+
            }
        }
        
        if not results:
            return summary
            
        total_score = 0.0
        
        for module in results.values():
            summary['total_functions'] += len(module.functions)
            summary['total_classes'] += len(module.classes)
            
            # Count high complexity functions
            for func in module.functions:
                if func.complexity > self.complexity_threshold:
                    summary['high_complexity_functions'] += 1
                    
                # Categorize complexity
                if func.complexity <= 5:
                    summary['complexity_distribution']['low'] += 1
                elif func.complexity <= 10:
                    summary['complexity_distribution']['medium'] += 1
                elif func.complexity <= 20:
                    summary['complexity_distribution']['high'] += 1
                else:
                    summary['complexity_distribution']['very_high'] += 1
            
            # Count high complexity classes
            for cls in module.classes:
                if cls.complexity > self.complexity_threshold:
                    summary['high_complexity_classes'] += 1
            
            # Count low maintainability files
            if module.overall_metrics.maintainability_index < self.maintainability_threshold:
                summary['low_maintainability_files'] += 1
                
            total_score += module.complexity_score
        
        summary['average_complexity_score'] = total_score / len(results)
        
        return summary
    
    def find_complexity_issues(self, results: Dict[str, ModuleComplexity]) -> List[Dict]:
        """Find specific complexity issues that need attention."""
        issues = []
        
        for file_path, module in results.items():
            # Check functions
            for func in module.functions:
                if func.complexity > self.complexity_threshold:
                    issues.append({
                        'type': 'high_complexity_function',
                        'file': file_path,
                        'name': func.name,
                        'line': func.lineno,
                        'complexity': func.complexity,
                        'threshold': self.complexity_threshold,
                        'severity': 'high' if func.complexity > 20 else 'medium'
                    })
            
            # Check classes
            for cls in module.classes:
                if cls.complexity > self.complexity_threshold:
                    issues.append({
                        'type': 'high_complexity_class',
                        'file': file_path,
                        'name': cls.name,
                        'line': cls.lineno,
                        'complexity': cls.complexity,
                        'threshold': self.complexity_threshold,
                        'severity': 'high' if cls.complexity > 20 else 'medium'
                    })
            
            # Check maintainability
            if module.overall_metrics.maintainability_index < self.maintainability_threshold:
                issues.append({
                    'type': 'low_maintainability',
                    'file': file_path,
                    'maintainability_index': module.overall_metrics.maintainability_index,
                    'threshold': self.maintainability_threshold,
                    'severity': 'high' if module.overall_metrics.maintainability_index < 40 else 'medium'
                })
        
        # Sort by severity and complexity
        issues.sort(key=lambda x: (
            {'high': 0, 'medium': 1, 'low': 2}[x['severity']],
            -x.get('complexity', 0)
        ))
        
        return issues