#!/usr/bin/env python3
"""
Enhanced Function Usage Analyzer for Python repositories.
Analyzes function usage patterns, dependencies, and identifies truly unused functions.
Works around syntax errors to provide accurate dependency mapping.
"""

import ast
import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple, Any, Optional
import importlib.util
import inspect
import traceback

class EnhancedFunctionUsageAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.function_definitions = defaultdict(dict)  # file -> {func_name -> func_info}
        self.function_calls = defaultdict(set)  # func_name -> set of calling_files
        self.function_imports = defaultdict(set)  # func_name -> set of importing_files
        self.class_methods = defaultdict(dict)  # file -> {class_name -> {method_name -> method_info}}
        self.dependency_graph = defaultdict(set)  # file -> set of dependencies
        self.usage_patterns = defaultdict(list)  # func_name -> list of usage patterns
        self.syntax_errors = defaultdict(list)
        self.import_mappings = defaultdict(dict)  # file -> {import_name -> actual_name}
        self.exported_functions = set()  # functions that are explicitly exported
        self.test_files = set()  # test files that might use functions
        self.config_files = set()  # config files that might reference functions
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
        return python_files
    
    def is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        test_patterns = ['test_', '_test', 'tests/', 'test/']
        return any(pattern in str(file_path) for pattern in test_patterns)
    
    def is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        config_patterns = ['config', 'settings', 'conf', 'setup']
        return any(pattern in str(file_path) for pattern in config_patterns)
    
    def extract_function_definitions_robust(self, file_path: Path) -> Dict[str, Dict]:
        """Extract function definitions using multiple parsing strategies."""
        functions = {}
        
        try:
            # Try standard AST parsing first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                functions = self._extract_functions_from_ast(tree, file_path)
            except SyntaxError as e:
                # Fallback to regex-based extraction for syntax errors
                functions = self._extract_functions_with_regex(content, file_path, e)
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"File read error: {e}")
            
        return functions
    
    def _extract_functions_from_ast(self, tree: ast.AST, file_path: Path) -> Dict[str, Dict]:
        """Extract functions from AST."""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'file': str(file_path),
                    'lineno': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'has_docstring': ast.get_docstring(node) is not None,
                    'body_lines': len(node.body) if node.body else 0
                }
                functions[node.name] = func_info
                
            elif isinstance(node, ast.ClassDef):
                # Extract methods from classes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = f"{node.name}.{item.name}"
                        func_info = {
                            'name': method_name,
                            'file': str(file_path),
                            'lineno': item.lineno,
                            'class_name': node.name,
                            'args': [arg.arg for arg in item.args.args],
                            'decorators': [self._get_decorator_name(d) for d in item.decorator_list],
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'has_docstring': ast.get_docstring(item) is not None,
                            'body_lines': len(item.body) if item.body else 0
                        }
                        functions[method_name] = func_info
                        
        return functions
    
    def _extract_functions_with_regex(self, content: str, file_path: Path, syntax_error: SyntaxError) -> Dict[str, Dict]:
        """Extract function definitions using regex when AST parsing fails."""
        functions = {}
        
        # Pattern to match function definitions
        func_pattern = r'^(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
        class_pattern = r'^class\s+(\w+)\s*[\(:]'
        method_pattern = r'^\s+(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
        
        lines = content.split('\n')
        current_class = None
        
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
                func_info = {
                    'name': func_name,
                    'file': str(file_path),
                    'lineno': i,
                    'args': self._extract_args_from_line(line),
                    'decorators': [],
                    'is_async': 'async' in line,
                    'has_docstring': False,
                    'body_lines': 0,
                    'extracted_with_regex': True
                }
                functions[func_name] = func_info
                
            # Check for method definition (indented)
            elif current_class and re.match(method_pattern, line):
                method_name = func_match.group(1) if func_match else None
                if method_name:
                    full_name = f"{current_class}.{method_name}"
                    func_info = {
                        'name': full_name,
                        'file': str(file_path),
                        'lineno': i,
                        'class_name': current_class,
                        'args': self._extract_args_from_line(line),
                        'decorators': [],
                        'is_async': 'async' in line,
                        'has_docstring': False,
                        'body_lines': 0,
                        'extracted_with_regex': True
                    }
                    functions[full_name] = func_info
                    
        return functions
    
    def _extract_args_from_line(self, line: str) -> List[str]:
        """Extract function arguments from a function definition line."""
        # Simple regex to extract arguments
        args_match = re.search(r'\(([^)]*)\)', line)
        if args_match:
            args_str = args_match.group(1)
            # Split by comma and extract argument names
            args = []
            for arg in args_str.split(','):
                arg = arg.strip()
                if arg:
                    # Remove type hints and default values
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
    
    def extract_function_calls_robust(self, file_path: Path) -> Set[str]:
        """Extract function calls using multiple strategies."""
        calls = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                calls.update(self._extract_calls_from_ast(tree))
            except SyntaxError:
                # Fallback to regex-based extraction
                calls.update(self._extract_calls_with_regex(content))
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"Call extraction error: {e}")
            
        return calls
    
    def _extract_calls_from_ast(self, tree: ast.AST) -> Set[str]:
        """Extract function calls from AST."""
        calls = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(f"{node.func.value.id}.{node.func.attr}")
                    
        return calls
    
    def _extract_calls_with_regex(self, content: str) -> Set[str]:
        """Extract function calls using regex."""
        calls = set()
        
        # Pattern to match function calls
        call_patterns = [
            r'(\w+)\s*\(',  # func(
            r'(\w+)\.(\w+)\s*\(',  # obj.method(
            r'(\w+)\s*\[',  # func[
            r'(\w+)\.(\w+)\s*\[',  # obj.method[
        ]
        
        for pattern in call_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        calls.add(f"{match[0]}.{match[1]}")
                    else:
                        calls.add(match[0])
                else:
                    calls.add(match)
                    
        return calls
    
    def extract_imports_robust(self, file_path: Path) -> Dict[str, str]:
        """Extract imports using multiple strategies."""
        imports = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                imports.update(self._extract_imports_from_ast(tree))
            except SyntaxError:
                # Fallback to regex-based extraction
                imports.update(self._extract_imports_with_regex(content))
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"Import extraction error: {e}")
            
        return imports
    
    def _extract_imports_from_ast(self, tree: ast.AST) -> Dict[str, str]:
        """Extract imports from AST."""
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        imports[alias.asname or alias.name] = full_name
                        
        return imports
    
    def _extract_imports_with_regex(self, content: str) -> Dict[str, str]:
        """Extract imports using regex."""
        imports = {}
        
        # Patterns for different import types
        import_patterns = [
            (r'import\s+(\w+)', r'\1'),  # import module
            (r'import\s+(\w+)\s+as\s+(\w+)', r'\2:\1'),  # import module as alias
            (r'from\s+(\w+)\s+import\s+(\w+)', r'\2:\1.\2'),  # from module import name
            (r'from\s+(\w+)\s+import\s+(\w+)\s+as\s+(\w+)', r'\3:\1.\2'),  # from module import name as alias
        ]
        
        for pattern, replacement in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        imports[match[1]] = match[0]
                    elif len(match) == 3:
                        imports[match[2]] = f"{match[0]}.{match[1]}"
                else:
                    imports[match] = match
                    
        return imports
    
    def analyze_function_usage(self):
        """Analyze function usage patterns and dependencies."""
        print("🔍 Analyzing function usage patterns...")
        
        # Build dependency graph
        for file_path, imports in self.import_mappings.items():
            for alias, full_name in imports.items():
                # Check if imported function is defined somewhere
                for def_file, functions in self.function_definitions.items():
                    for func_name in functions:
                        if func_name == full_name or func_name.endswith(f".{full_name}"):
                            self.dependency_graph[def_file].add(file_path)
        
        # Analyze usage patterns
        for func_name in self.function_definitions:
            usage_patterns = []
            
            # Check direct calls
            if func_name in self.function_calls:
                for calling_file in self.function_calls[func_name]:
                    usage_patterns.append({
                        'type': 'direct_call',
                        'file': calling_file,
                        'pattern': 'function_call'
                    })
            
            # Check imports
            for file_path, imports in self.import_mappings.items():
                for alias, full_name in imports.items():
                    if func_name == full_name or func_name.endswith(f".{full_name}"):
                        usage_patterns.append({
                            'type': 'import',
                            'file': file_path,
                            'pattern': 'imported_function'
                        })
            
            # Check for test usage
            for test_file in self.test_files:
                if any(test_file in pattern['file'] for pattern in usage_patterns):
                    usage_patterns.append({
                        'type': 'test_usage',
                        'file': str(test_file),
                        'pattern': 'test_function'
                    })
            
            self.usage_patterns[func_name] = usage_patterns
    
    def identify_truly_unused_functions(self) -> Dict[str, List[str]]:
        """Identify functions that are truly unused (no dependencies)."""
        print("🔍 Identifying truly unused functions...")
        
        unused_functions = defaultdict(list)
        potentially_used = set()
        
        for func_name, usage_patterns in self.usage_patterns.items():
            if not usage_patterns:
                # No usage patterns found
                file_path = self._get_function_file(func_name)
                if file_path:
                    unused_functions[file_path].append(func_name)
            else:
                # Check if usage is meaningful
                meaningful_usage = False
                for pattern in usage_patterns:
                    if pattern['type'] in ['direct_call', 'import']:
                        meaningful_usage = True
                        break
                
                if not meaningful_usage:
                    file_path = self._get_function_file(func_name)
                    if file_path:
                        unused_functions[file_path].append(func_name)
                else:
                    potentially_used.add(func_name)
        
        return unused_functions
    
    def _get_function_file(self, func_name: str) -> Optional[str]:
        """Get the file where a function is defined."""
        for file_path, functions in self.function_definitions.items():
            if func_name in functions:
                return file_path
        return None
    
    def analyze_dependencies(self) -> Dict[str, Dict]:
        """Analyze dependencies between functions and files."""
        print("🔍 Analyzing function dependencies...")
        
        dependencies = {}
        
        for func_name, usage_patterns in self.usage_patterns.items():
            deps = {
                'direct_calls': [],
                'imports': [],
                'test_usage': [],
                'config_usage': [],
                'hidden_dependencies': []
            }
            
            for pattern in usage_patterns:
                if pattern['type'] == 'direct_call':
                    deps['direct_calls'].append(pattern['file'])
                elif pattern['type'] == 'import':
                    deps['imports'].append(pattern['file'])
                elif pattern['type'] == 'test_usage':
                    deps['test_usage'].append(pattern['file'])
                elif pattern['type'] == 'config_usage':
                    deps['config_usage'].append(pattern['file'])
            
            # Check for hidden dependencies (functions called by imported functions)
            hidden_deps = self._find_hidden_dependencies(func_name)
            deps['hidden_dependencies'] = hidden_deps
            
            dependencies[func_name] = deps
        
        return dependencies
    
    def _find_hidden_dependencies(self, func_name: str) -> List[str]:
        """Find hidden dependencies through imported functions."""
        hidden_deps = []
        
        # This is a simplified approach - in practice, you'd need more sophisticated analysis
        for file_path, imports in self.import_mappings.items():
            for alias, full_name in imports.items():
                if func_name in full_name:
                    # Check if this import is used in function calls
                    if file_path in self.function_calls:
                        hidden_deps.append(str(file_path))
        
        return hidden_deps
    
    def generate_safety_report(self) -> Dict[str, Any]:
        """Generate a safety report for function removal."""
        print("🔍 Generating safety report...")
        
        safety_report = {
            'high_risk_functions': [],
            'medium_risk_functions': [],
            'low_risk_functions': [],
            'safe_to_remove': [],
            'requires_further_analysis': []
        }
        
        dependencies = self.analyze_dependencies()
        
        for func_name, deps in dependencies.items():
            risk_score = self._calculate_risk_score(deps)
            
            if risk_score >= 8:
                safety_report['high_risk_functions'].append({
                    'function': func_name,
                    'risk_score': risk_score,
                    'dependencies': deps,
                    'reason': 'High dependency count or critical usage patterns'
                })
            elif risk_score >= 5:
                safety_report['medium_risk_functions'].append({
                    'function': func_name,
                    'risk_score': risk_score,
                    'dependencies': deps,
                    'reason': 'Moderate dependencies or unclear usage patterns'
                })
            elif risk_score >= 2:
                safety_report['low_risk_functions'].append({
                    'function': func_name,
                    'risk_score': risk_score,
                    'dependencies': deps,
                    'reason': 'Low dependencies but some usage detected'
                })
            elif risk_score == 0:
                safety_report['safe_to_remove'].append({
                    'function': func_name,
                    'risk_score': risk_score,
                    'dependencies': deps,
                    'reason': 'No dependencies detected'
                })
            else:
                safety_report['requires_further_analysis'].append({
                    'function': func_name,
                    'risk_score': risk_score,
                    'dependencies': deps,
                    'reason': 'Insufficient information for risk assessment'
                })
        
        return safety_report
    
    def _calculate_risk_score(self, deps: Dict) -> int:
        """Calculate risk score for function removal."""
        score = 0
        
        # Direct calls are high risk
        score += len(deps['direct_calls']) * 3
        
        # Imports are medium risk
        score += len(deps['imports']) * 2
        
        # Test usage is low risk
        score += len(deps['test_usage']) * 1
        
        # Config usage is medium risk
        score += len(deps['config_usage']) * 2
        
        # Hidden dependencies are high risk
        score += len(deps['hidden_dependencies']) * 3
        
        return score
    
    def analyze_all_files(self):
        """Analyze all Python files in the repository."""
        print(f"🔍 Analyzing Python files in: {self.root_dir}")
        
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files to analyze...")
        
        for i, file_path in enumerate(python_files):
            if i % 20 == 0:
                print(f"Processing file {i+1}/{len(python_files)}...")
            
            # Categorize file
            if self.is_test_file(file_path):
                self.test_files.add(str(file_path))
            elif self.is_config_file(file_path):
                self.config_files.add(str(file_path))
            
            # Extract function definitions
            functions = self.extract_function_definitions_robust(file_path)
            if functions:
                self.function_definitions[str(file_path)] = functions
            
            # Extract function calls
            calls = self.extract_function_calls_robust(file_path)
            for call in calls:
                self.function_calls[call].add(str(file_path))
            
            # Extract imports
            imports = self.extract_imports_robust(file_path)
            if imports:
                self.import_mappings[str(file_path)] = imports
        
        print("✅ File analysis complete!")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "summary": {
                "total_files_analyzed": len(self.function_definitions),
                "total_functions_found": sum(len(funcs) for funcs in self.function_definitions.values()),
                "files_with_syntax_errors": len(self.syntax_errors),
                "test_files": len(self.test_files),
                "config_files": len(self.config_files)
            },
            "function_definitions": {
                str(k): v for k, v in self.function_definitions.items()
            },
            "function_calls": {
                k: list(v) for k, v in self.function_calls.items()
            },
            "import_mappings": {
                str(k): v for k, v in self.import_mappings.items()
            },
            "usage_patterns": {
                k: v for k, v in self.usage_patterns.items()
            },
            "syntax_errors": {
                str(k): v for k, v in self.syntax_errors.items()
            },
            "safety_report": self.generate_safety_report()
        }
        
        return report
    
    def save_report(self, output_path: str = "enhanced_function_usage_report.json"):
        """Save report as JSON."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Enhanced function usage report saved to {output_path}")
    
    def print_summary(self):
        """Print analysis summary."""
        report = self.generate_report()
        summary = report["summary"]
        safety_report = report["safety_report"]
        
        print(f"\n{'='*80}")
        print(f"ENHANCED FUNCTION USAGE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"📁 Total files analyzed: {summary['total_files_analyzed']}")
        print(f"🔧 Total functions found: {summary['total_functions_found']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        print(f"🧪 Test files: {summary['test_files']}")
        print(f"⚙️  Config files: {summary['config_files']}")
        
        print(f"\n🛡️  SAFETY ASSESSMENT:")
        print(f"   • High risk functions: {len(safety_report['high_risk_functions'])}")
        print(f"   • Medium risk functions: {len(safety_report['medium_risk_functions'])}")
        print(f"   • Low risk functions: {len(safety_report['low_risk_functions'])}")
        print(f"   • Safe to remove: {len(safety_report['safe_to_remove'])}")
        print(f"   • Requires further analysis: {len(safety_report['requires_further_analysis'])}")
        
        if safety_report['safe_to_remove']:
            print(f"\n✅ SAFE TO REMOVE (Top 10):")
            for i, func_info in enumerate(safety_report['safe_to_remove'][:10], 1):
                print(f"   {i:2d}. {func_info['function']} (Risk: {func_info['risk_score']})")
        
        if safety_report['high_risk_functions']:
            print(f"\n🚨 HIGH RISK FUNCTIONS (Top 10):")
            for i, func_info in enumerate(safety_report['high_risk_functions'][:10], 1):
                print(f"   {i:2d}. {func_info['function']} (Risk: {func_info['risk_score']})")
                print(f"       Reason: {func_info['reason']}")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Enhanced function usage analysis for: {root_dir}")
    
    analyzer = EnhancedFunctionUsageAnalyzer(root_dir)
    
    # Analyze all files
    analyzer.analyze_all_files()
    
    # Analyze function usage
    analyzer.analyze_function_usage()
    
    # Generate outputs
    analyzer.save_report()
    
    # Print summary
    analyzer.print_summary()

if __name__ == "__main__":
    main()