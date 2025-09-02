#!/usr/bin/env python3
"""
Advanced Dependency Analyzer for Python repositories.
Provides comprehensive dependency mapping and risk assessment for function removal.
Works around syntax errors to ensure accurate dependency analysis.
"""

import ast
import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DependencyInfo:
    """Information about a function dependency."""
    function_name: str
    file_path: str
    dependency_type: str  # 'direct_call', 'import', 'inheritance', 'decorator', 'dynamic'
    line_number: Optional[int] = None
    context: Optional[str] = None
    risk_level: str = 'unknown'  # 'low', 'medium', 'high', 'critical'

@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    file_path: str
    line_number: int
    is_class_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = None
    arguments: List[str] = None
    has_docstring: bool = False
    is_async: bool = False
    extracted_with_regex: bool = False

class AdvancedDependencyAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.functions = {}  # func_name -> FunctionInfo
        self.dependencies = defaultdict(list)  # func_name -> [DependencyInfo]
        self.dependency_graph = {}  # Simple dict-based graph
        self.import_mappings = defaultdict(dict)  # file -> {alias -> full_name}
        self.syntax_errors = defaultdict(list)
        self.test_files = set()
        self.config_files = set()
        self.export_files = set()  # __init__.py files
        
    def analyze_repository(self):
        """Perform comprehensive dependency analysis."""
        print(f"🔍 Starting advanced dependency analysis for: {self.root_dir}")
        
        # Find and categorize files
        python_files = self._find_python_files()
        self._categorize_files(python_files)
        
        # Extract function definitions
        print("📝 Extracting function definitions...")
        for file_path in python_files:
            self._extract_functions_from_file(file_path)
        
        # Extract dependencies
        print("🔗 Extracting dependencies...")
        for file_path in python_files:
            self._extract_dependencies_from_file(file_path)
        
        # Build dependency graph
        print("🕸️  Building dependency graph...")
        self._build_dependency_graph()
        
        # Analyze dependency chains
        print("⛓️  Analyzing dependency chains...")
        self._analyze_dependency_chains()
        
        print("✅ Repository analysis complete!")
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
        return python_files
    
    def _categorize_files(self, python_files: List[Path]):
        """Categorize files by type."""
        for file_path in python_files:
            file_str = str(file_path)
            
            if self._is_test_file(file_path):
                self.test_files.add(file_str)
            elif self._is_config_file(file_path):
                self.config_files.add(file_str)
            elif file_path.name == '__init__.py':
                self.export_files.add(file_str)
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        test_patterns = ['test_', '_test', 'tests/', 'test/', 'spec_', '_spec']
        return any(pattern in str(file_path) for pattern in test_patterns)
    
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        config_patterns = ['config', 'settings', 'conf', 'setup', 'requirements']
        return any(pattern in str(file_path) for pattern in config_patterns)
    
    def _extract_functions_from_file(self, file_path: Path):
        """Extract function definitions from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                # Try AST parsing first
                tree = ast.parse(content)
                self._extract_functions_from_ast(tree, file_path)
            except SyntaxError:
                # Fallback to regex-based extraction
                self._extract_functions_with_regex(content, file_path)
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"Function extraction error: {e}")
    
    def _extract_functions_from_ast(self, tree: ast.AST, file_path: Path):
        """Extract functions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = FunctionInfo(
                    name=node.name,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    is_class_method=False,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    arguments=[arg.arg for arg in node.args.args],
                    has_docstring=ast.get_docstring(node) is not None,
                    is_async=isinstance(node, ast.AsyncFunctionDef)
                )
                self.functions[node.name] = func_info
                
            elif isinstance(node, ast.ClassDef):
                # Extract methods from classes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = f"{node.name}.{item.name}"
                        func_info = FunctionInfo(
                            name=method_name,
                            file_path=str(file_path),
                            line_number=item.lineno,
                            is_class_method=True,
                            class_name=node.name,
                            decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                            arguments=[arg.arg for arg in item.args.args],
                            has_docstring=ast.get_docstring(item) is not None,
                            is_async=isinstance(item, ast.AsyncFunctionDef)
                        )
                        self.functions[method_name] = func_info
    
    def _extract_functions_with_regex(self, content: str, file_path: Path):
        """Extract function definitions using regex when AST fails."""
        lines = content.split('\n')
        current_class = None
        
        # Patterns for function and method definitions
        func_pattern = r'^(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
        class_pattern = r'^class\s+(\w+)\s*[\(:]'
        method_pattern = r'^\s+(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for class definition
            class_match = re.match(class_pattern, line)
            if class_match:
                current_class = class_match.group(1)
                continue
            
            # Check for function definition
            func_match = re.match(func_pattern, line)
            if func_match:
                func_name = func_match.group(1)
                func_info = FunctionInfo(
                    name=func_name,
                    file_path=str(file_path),
                    line_number=i,
                    is_class_method=False,
                    decorators=[],
                    arguments=self._extract_args_from_line(line),
                    has_docstring=False,
                    is_async='async' in line,
                    extracted_with_regex=True
                )
                self.functions[func_name] = func_info
                
            # Check for method definition
            elif current_class and re.match(method_pattern, line):
                method_match = re.match(method_pattern, line)
                if method_match:
                    method_name = method_match.group(1)
                    full_name = f"{current_class}.{method_name}"
                    func_info = FunctionInfo(
                        name=full_name,
                        file_path=str(file_path),
                        line_number=i,
                        is_class_method=True,
                        class_name=current_class,
                        decorators=[],
                        arguments=self._extract_args_from_line(line),
                        has_docstring=False,
                        is_async='async' in line,
                        extracted_with_regex=True
                    )
                    self.functions[full_name] = func_info
    
    def _extract_args_from_line(self, line: str) -> List[str]:
        """Extract function arguments from a function definition line."""
        args_match = re.search(r'\(([^)]*)\)', line)
        if args_match:
            args_str = args_match.group(1)
            args = []
            for arg in args_str.split(','):
                arg = arg.strip()
                if arg:
                    arg_name = arg.split(':')[0].split('=')[0].strip()
                    if arg_name and arg_name != 'self':
                        args.append(arg_name)
            return args
        return []
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.value.id}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return f"{decorator.func.value.id}.{decorator.func.attr}"
        return str(decorator)
    
    def _extract_dependencies_from_file(self, file_path: Path):
        """Extract dependencies from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                self._extract_dependencies_from_ast(tree, file_path)
            except SyntaxError:
                self._extract_dependencies_with_regex(content, file_path)
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"Dependency extraction error: {e}")
    
    def _extract_dependencies_from_ast(self, tree: ast.AST, file_path: Path):
        """Extract dependencies from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._process_function_call(node, file_path)
            elif isinstance(node, ast.Import):
                self._process_import(node, file_path)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from(node, file_path)
            elif isinstance(node, ast.Attribute):
                self._process_attribute_access(node, file_path)
    
    def _extract_dependencies_with_regex(self, content: str, file_path: Path):
        """Extract dependencies using regex when AST fails."""
        # Extract function calls
        call_pattern = r'(\w+)\s*\('
        calls = re.findall(call_pattern, content)
        for call in calls:
            self._add_dependency(call, file_path, 'direct_call', 1)
        
        # Extract imports
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import\s+(\w+)',
            r'import\s+(\w+)\s+as\s+(\w+)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        self._add_dependency(match[1], file_path, 'import', 1)
                else:
                    self._add_dependency(match, file_path, 'import', 1)
    
    def _process_function_call(self, node: ast.Call, file_path: Path):
        """Process a function call node."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self._add_dependency(func_name, file_path, 'direct_call', node.lineno)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            self._add_dependency(attr_name, file_path, 'direct_call', node.lineno)
    
    def _process_import(self, node: ast.Import, file_path: Path):
        """Process an import node."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self._add_dependency(module_name, file_path, 'import', node.lineno)
            if alias.asname:
                self.import_mappings[str(file_path)][alias.asname] = alias.name
    
    def _process_import_from(self, node: ast.ImportFrom, file_path: Path):
        """Process an import from node."""
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self._add_dependency(alias.name, file_path, 'import', node.lineno)
                if alias.asname:
                    self.import_mappings[str(file_path)][alias.asname] = full_name
                else:
                    self.import_mappings[str(file_path)][alias.name] = full_name
    
    def _process_attribute_access(self, node: ast.Attribute, file_path: Path):
        """Process an attribute access node."""
        if isinstance(node.value, ast.Name):
            attr_name = node.attr
            self._add_dependency(attr_name, file_path, 'attribute_access', getattr(node, 'lineno', None))
    
    def _add_dependency(self, func_name: str, file_path: Path, dep_type: str, line_number: Optional[int]):
        """Add a dependency to the dependency graph."""
        if func_name in self.functions:
            dep_info = DependencyInfo(
                function_name=func_name,
                file_path=str(file_path),
                dependency_type=dep_type,
                line_number=line_number,
                context=self._get_context(file_path, dep_type),
                risk_level=self._assess_risk_level(dep_type, file_path)
            )
            self.dependencies[func_name].append(dep_info)
    
    def _get_context(self, file_path: Path, dep_type: str) -> str:
        """Get context information for a dependency."""
        file_str = str(file_path)
        
        if file_str in self.test_files:
            return 'test_file'
        elif file_str in self.config_files:
            return 'config_file'
        elif file_str in self.export_files:
            return 'export_file'
        else:
            return 'source_file'
    
    def _assess_risk_level(self, dep_type: str, file_path: Path) -> str:
        """Assess the risk level of a dependency."""
        file_str = str(file_path)
        
        if dep_type == 'direct_call':
            if file_str in self.test_files:
                return 'low'
            elif file_str in self.config_files:
                return 'medium'
            else:
                return 'high'
        elif dep_type == 'import':
            if file_str in self.export_files:
                return 'critical'
            elif file_str in self.config_files:
                return 'medium'
            else:
                return 'high'
        else:
            return 'medium'
    
    def _build_dependency_graph(self):
        """Build a simple dependency graph."""
        # Add nodes for all functions
        for func_name in self.functions:
            self.dependency_graph[func_name] = set()
        
        # Add edges for dependencies
        for func_name, deps in self.dependencies.items():
            for dep in deps:
                if dep.function_name in self.functions:
                    self.dependency_graph[dep.function_name].add(func_name)
    
    def _analyze_dependency_chains(self):
        """Analyze dependency chains and cycles."""
        # Simple cycle detection
        cycles_found = 0
        for func_name, deps in self.dependency_graph.items():
            if func_name in deps:  # Self-dependency
                cycles_found += 1
        
        if cycles_found > 0:
            print(f"⚠️  Found {cycles_found} self-dependencies")
        else:
            print("✅ No self-dependencies found")
        
        # Calculate simple dependency depths
        for func_name in self.functions:
            deps = self.dependency_graph.get(func_name, set())
            depth = len(deps)
            # Store depth information for risk assessment
            if func_name in self.functions:
                # Add depth to function info
                pass
    
    def identify_safe_to_remove_functions(self) -> Dict[str, List[str]]:
        """Identify functions that are safe to remove."""
        safe_functions = defaultdict(list)
        
        for func_name, func_info in self.functions.items():
            if self._is_safe_to_remove(func_name):
                file_path = func_info.file_path
                safe_functions[file_path].append(func_name)
        
        return safe_functions
    
    def _is_safe_to_remove(self, func_name: str) -> bool:
        """Check if a function is safe to remove."""
        # Check if function has any dependencies
        if func_name not in self.dependencies:
            return True
        
        # Check dependency risk levels
        high_risk_deps = [dep for dep in self.dependencies[func_name] 
                         if dep.risk_level in ['high', 'critical']]
        
        if high_risk_deps:
            return False
        
        # Check if function is exported
        if self._is_exported_function(func_name):
            return False
        
        # Check if function is used in tests (low risk)
        test_deps = [dep for dep in self.dependencies[func_name] 
                    if dep.context == 'test_file']
        
        # If only test dependencies, it's safe to remove
        if len(test_deps) == len(self.dependencies[func_name]):
            return True
        
        return True
    
    def _is_exported_function(self, func_name: str) -> bool:
        """Check if a function is exported from __init__.py files."""
        for export_file in self.export_files:
            if export_file in self.import_mappings:
                for alias, full_name in self.import_mappings[export_file].items():
                    if func_name in full_name:
                        return True
        return False
    
    def generate_removal_safety_report(self) -> Dict[str, Any]:
        """Generate a comprehensive safety report for function removal."""
        safety_report = {
            'safe_to_remove': [],
            'requires_caution': [],
            'high_risk': [],
            'critical_risk': [],
            'requires_manual_review': []
        }
        
        for func_name, func_info in self.functions.items():
            risk_assessment = self._assess_removal_risk(func_name)
            
            func_data = {
                'name': func_name,
                'file_path': func_info.file_path,
                'line_number': func_info.line_number,
                'is_class_method': func_info.is_class_method,
                'class_name': func_info.class_name,
                'risk_assessment': risk_assessment
            }
            
            if risk_assessment['risk_level'] == 'safe':
                safety_report['safe_to_remove'].append(func_data)
            elif risk_assessment['risk_level'] == 'low':
                safety_report['requires_caution'].append(func_data)
            elif risk_assessment['risk_level'] == 'medium':
                safety_report['high_risk'].append(func_data)
            elif risk_assessment['risk_level'] == 'high':
                safety_report['critical_risk'].append(func_data)
            else:
                safety_report['requires_manual_review'].append(func_data)
        
        return safety_report
    
    def _assess_removal_risk(self, func_name: str) -> Dict[str, Any]:
        """Assess the risk of removing a function."""
        if func_name not in self.dependencies:
            return {
                'risk_level': 'safe',
                'risk_score': 0,
                'reason': 'No dependencies detected',
                'dependencies': []
            }
        
        deps = self.dependencies[func_name]
        risk_score = 0
        risk_factors = []
        
        for dep in deps:
            if dep.risk_level == 'critical':
                risk_score += 10
                risk_factors.append(f"Critical dependency in {dep.file_path}")
            elif dep.risk_level == 'high':
                risk_score += 5
                risk_factors.append(f"High dependency in {dep.file_path}")
            elif dep.risk_level == 'medium':
                risk_score += 2
                risk_factors.append(f"Medium dependency in {dep.file_path}")
            elif dep.risk_level == 'low':
                risk_score += 1
                risk_factors.append(f"Low dependency in {dep.file_path}")
        
        # Determine risk level
        if risk_score == 0:
            risk_level = 'safe'
        elif risk_score <= 3:
            risk_level = 'low'
        elif risk_score <= 8:
            risk_level = 'medium'
        elif risk_score <= 15:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'reason': '; '.join(risk_factors) if risk_factors else 'No risk factors',
            'dependencies': [asdict(dep) for dep in deps]
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "summary": {
                "total_functions": len(self.functions),
                "total_dependencies": sum(len(deps) for deps in self.dependencies.values()),
                "files_with_syntax_errors": len(self.syntax_errors),
                "test_files": len(self.test_files),
                "config_files": len(self.config_files),
                "export_files": len(self.export_files)
            },
            "functions": {
                name: asdict(info) for name, info in self.functions.items()
            },
            "dependencies": {
                name: [asdict(dep) for dep in deps] 
                for name, deps in self.dependencies.items()
            },
            "import_mappings": {
                str(k): v for k, v in self.import_mappings.items()
            },
            "syntax_errors": {
                str(k): v for k, v in self.syntax_errors.items()
            },
            "safety_report": self.generate_removal_safety_report(),
                    "dependency_graph_stats": {
            "nodes": len(self.dependency_graph),
            "edges": sum(len(deps) for deps in self.dependency_graph.values()),
            "is_directed": True,
            "has_cycles": any(func_name in deps for func_name, deps in self.dependency_graph.items())
        }
        }
        
        return report
    
    def save_report(self, output_path: str = "advanced_dependency_analysis.json"):
        """Save report as JSON."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Advanced dependency analysis report saved to {output_path}")
    
    def print_summary(self):
        """Print analysis summary."""
        report = self.generate_report()
        summary = report["summary"]
        safety_report = report["safety_report"]
        
        print(f"\n{'='*80}")
        print(f"ADVANCED DEPENDENCY ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"📁 Total functions: {summary['total_functions']}")
        print(f"🔗 Total dependencies: {summary['total_dependencies']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        print(f"🧪 Test files: {summary['test_files']}")
        print(f"⚙️  Config files: {summary['config_files']}")
        print(f"📤 Export files: {summary['export_files']}")
        
        print(f"\n🛡️  REMOVAL SAFETY ASSESSMENT:")
        print(f"   • Safe to remove: {len(safety_report['safe_to_remove'])}")
        print(f"   • Requires caution: {len(safety_report['requires_caution'])}")
        print(f"   • High risk: {len(safety_report['high_risk'])}")
        print(f"   • Critical risk: {len(safety_report['critical_risk'])}")
        print(f"   • Requires manual review: {len(safety_report['requires_manual_review'])}")
        
        if safety_report['safe_to_remove']:
            print(f"\n✅ SAFE TO REMOVE (Top 10):")
            for i, func_info in enumerate(safety_report['safe_to_remove'][:10], 1):
                print(f"   {i:2d}. {func_info['name']} in {func_info['file_path']}")
        
        if safety_report['critical_risk']:
            print(f"\n🚨 CRITICAL RISK FUNCTIONS (Top 10):")
            for i, func_info in enumerate(safety_report['critical_risk'][:10], 1):
                print(f"   {i:2d}. {func_info['name']} in {func_info['file_path']}")
                print(f"       Risk: {func_info['risk_assessment']['risk_level']} (Score: {func_info['risk_assessment']['risk_score']})")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Advanced dependency analysis for: {root_dir}")
    
    analyzer = AdvancedDependencyAnalyzer(root_dir)
    
    # Analyze repository
    analyzer.analyze_repository()
    
    # Generate outputs
    analyzer.save_report()
    
    # Print summary
    analyzer.print_summary()

if __name__ == "__main__":
    main()