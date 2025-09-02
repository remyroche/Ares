#!/usr/bin/env python3
"""
Function Usage Validator for Python repositories.
Validates function usage patterns and identifies truly unused functions.
Works around syntax errors to provide accurate validation.
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

class FunctionUsageValidator:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.function_definitions = {}  # func_name -> FunctionDefinition
        self.function_usage = defaultdict(list)  # func_name -> [UsageInfo]
        self.import_statements = defaultdict(list)  # file -> [ImportInfo]
        self.export_statements = defaultdict(list)  # file -> [ExportInfo]
        self.syntax_errors = defaultdict(list)
        self.validation_results = {}
        
    def validate_all_functions(self):
        """Perform comprehensive function usage validation."""
        print(f"🔍 Starting function usage validation for: {self.root_dir}")
        
        # Find all Python files
        python_files = self._find_python_files()
        print(f"Found {len(python_files)} Python files to validate...")
        
        # Extract function definitions
        print("📝 Extracting function definitions...")
        for file_path in python_files:
            self._extract_function_definitions(file_path)
        
        # Extract usage patterns
        print("🔍 Extracting usage patterns...")
        for file_path in python_files:
            self._extract_usage_patterns(file_path)
        
        # Extract import and export statements
        print("📥 Extracting import/export statements...")
        for file_path in python_files:
            self._extract_imports_exports(file_path)
        
        # Validate function usage
        print("✅ Validating function usage...")
        self._validate_function_usage()
        
        print("✅ Function usage validation complete!")
    
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
    
    def _extract_function_definitions(self, file_path: Path):
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
        """Extract function definitions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = FunctionDefinition(
                    name=node.name,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    is_class_method=False,
                    class_name=None,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    arguments=[arg.arg for arg in node.args.args],
                    has_docstring=ast.get_docstring(node) is not None,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    body_lines=len(node.body) if node.body else 0
                )
                self.function_definitions[node.name] = func_def
                
            elif isinstance(node, ast.ClassDef):
                # Extract methods from classes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = f"{node.name}.{item.name}"
                        func_def = FunctionDefinition(
                            name=method_name,
                            file_path=str(file_path),
                            line_number=item.lineno,
                            is_class_method=True,
                            class_name=node.name,
                            decorators=[self._get_decorator_name(d) for d in item.decorator_list],
                            arguments=[arg.arg for arg in item.args.args],
                            has_docstring=ast.get_docstring(item) is not None,
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            body_lines=len(item.body) if item.body else 0
                        )
                        self.function_definitions[method_name] = func_def
    
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
                func_def = FunctionDefinition(
                    name=func_name,
                    file_path=str(file_path),
                    line_number=i,
                    is_class_method=False,
                    class_name=None,
                    decorators=[],
                    arguments=self._extract_args_from_line(line),
                    has_docstring=False,
                    is_async='async' in line,
                    body_lines=0,
                    extracted_with_regex=True
                )
                self.function_definitions[func_name] = func_def
                
            # Check for method definition
            elif current_class and re.match(method_pattern, line):
                method_match = re.match(method_pattern, line)
                if method_match:
                    method_name = method_match.group(1)
                    full_name = f"{current_class}.{method_name}"
                    func_def = FunctionDefinition(
                        name=full_name,
                        file_path=str(file_path),
                        line_number=i,
                        is_class_method=True,
                        class_name=current_class,
                        decorators=[],
                        arguments=self._extract_args_from_line(line),
                        has_docstring=False,
                        is_async='async' in line,
                        body_lines=0,
                        extracted_with_regex=True
                    )
                    self.function_definitions[full_name] = func_def
    
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
    
    def _extract_usage_patterns(self, file_path: Path):
        """Extract function usage patterns from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                self._extract_usage_from_ast(tree, file_path)
            except SyntaxError:
                self._extract_usage_with_regex(content, file_path)
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"Usage extraction error: {e}")
    
    def _extract_usage_from_ast(self, tree: ast.AST, file_path: Path):
        """Extract usage patterns from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._process_function_call(node, file_path)
            elif isinstance(node, ast.Attribute):
                self._process_attribute_access(node, file_path)
            elif isinstance(node, ast.Name):
                self._process_name_usage(node, file_path)
    
    def _extract_usage_with_regex(self, content: str, file_path: Path):
        """Extract usage patterns using regex when AST fails."""
        # Extract function calls
        call_patterns = [
            r'(\w+)\s*\(',
            r'(\w+)\.(\w+)\s*\(',
            r'(\w+)\s*\[',
            r'(\w+)\.(\w+)\s*\['
        ]
        
        for pattern in call_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        usage_info = UsageInfo(
                            function_name=match[1],
                            file_path=str(file_path),
                            usage_type='attribute_access',
                            line_number=1,
                            context='regex_extraction'
                        )
                        self.function_usage[match[1]].append(usage_info)
                else:
                    usage_info = UsageInfo(
                        function_name=match,
                        file_path=str(file_path),
                        usage_type='function_call',
                        line_number=1,
                        context='regex_extraction'
                    )
                    self.function_usage[match].append(usage_info)
    
    def _process_function_call(self, node: ast.Call, file_path: Path):
        """Process a function call node."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            usage_info = UsageInfo(
                function_name=func_name,
                file_path=str(file_path),
                usage_type='function_call',
                line_number=node.lineno,
                context='ast_analysis'
            )
            self.function_usage[func_name].append(usage_info)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            usage_info = UsageInfo(
                function_name=attr_name,
                file_path=str(file_path),
                usage_type='method_call',
                line_number=node.lineno,
                context='ast_analysis'
            )
            self.function_usage[attr_name].append(usage_info)
    
    def _process_attribute_access(self, node: ast.Attribute, file_path: Path):
        """Process an attribute access node."""
        if isinstance(node.value, ast.Name):
            attr_name = node.attr
            usage_info = UsageInfo(
                function_name=attr_name,
                file_path=str(file_path),
                usage_type='attribute_access',
                line_number=getattr(node, 'lineno', 1),
                context='ast_analysis'
            )
            self.function_usage[attr_name].append(usage_info)
    
    def _process_name_usage(self, node: ast.Name, file_path: Path):
        """Process a name usage node."""
        # Check if this name is used in a context that suggests function usage
        if hasattr(node, 'ctx') and isinstance(node.ctx, ast.Load):
            func_name = node.id
            usage_info = UsageInfo(
                function_name=func_name,
                file_path=str(file_path),
                usage_type='name_reference',
                line_number=getattr(node, 'lineno', 1),
                context='ast_analysis'
            )
            self.function_usage[func_name].append(usage_info)
    
    def _extract_imports_exports(self, file_path: Path):
        """Extract import and export statements from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                self._extract_imports_exports_from_ast(tree, file_path)
            except SyntaxError:
                self._extract_imports_exports_with_regex(content, file_path)
                
        except Exception as e:
            self.syntax_errors[str(file_path)].append(f"Import/export extraction error: {e}")
    
    def _extract_imports_exports_from_ast(self, tree: ast.AST, file_path: Path):
        """Extract import and export statements from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = ImportInfo(
                        module_name=alias.name,
                        alias_name=alias.asname or alias.name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        import_type='import'
                    )
                    self.import_statements[str(file_path)].append(import_info)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        import_info = ImportInfo(
                            module_name=f"{node.module}.{alias.name}",
                            alias_name=alias.asname or alias.name,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            import_type='from_import'
                        )
                        self.import_statements[str(file_path)].append(import_info)
            
            elif isinstance(node, ast.Assign):
                # Check for exports (assignments to __all__)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            for item in node.value.elts:
                                if isinstance(item, ast.Str):
                                    export_info = ExportInfo(
                                        name=item.s,
                                        file_path=str(file_path),
                                        line_number=node.lineno,
                                        export_type='__all__'
                                    )
                                    self.export_statements[str(file_path)].append(export_info)
    
    def _extract_imports_exports_with_regex(self, content: str, file_path: Path):
        """Extract import and export statements using regex when AST fails."""
        # Extract imports
        import_patterns = [
            (r'import\s+(\w+)', r'\1'),
            (r'from\s+(\w+)\s+import\s+(\w+)', r'\1.\2'),
            (r'import\s+(\w+)\s+as\s+(\w+)', r'\1')
        ]
        
        for pattern, replacement in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        import_info = ImportInfo(
                            module_name=match[1],
                            alias_name=match[1],
                            file_path=str(file_path),
                            line_number=1,
                            import_type='regex_extraction'
                        )
                        self.import_statements[str(file_path)].append(import_info)
                else:
                    import_info = ImportInfo(
                        module_name=match,
                        alias_name=match,
                        file_path=str(file_path),
                        line_number=1,
                        import_type='regex_extraction'
                    )
                    self.import_statements[str(file_path)].append(import_info)
        
        # Extract exports
        export_pattern = r'__all__\s*=\s*\[([^\]]+)\]'
        export_matches = re.findall(export_pattern, content)
        for match in export_matches:
            exports = [exp.strip().strip("'\"") for exp in match.split(',')]
            for export in exports:
                if export:
                    export_info = ExportInfo(
                        name=export,
                        file_path=str(file_path),
                        line_number=1,
                        export_type='regex_extraction'
                    )
                    self.export_statements[str(file_path)].append(export_info)
    
    def _validate_function_usage(self):
        """Validate function usage patterns and identify truly unused functions."""
        for func_name, func_def in self.function_definitions.items():
            validation_result = self._validate_single_function(func_name, func_def)
            self.validation_results[func_name] = validation_result
    
    def _validate_single_function(self, func_name: str, func_def: 'FunctionDefinition') -> Dict[str, Any]:
        """Validate usage of a single function."""
        usage_patterns = self.function_usage.get(func_name, [])
        
        # Analyze usage patterns
        usage_analysis = self._analyze_usage_patterns(usage_patterns)
        
        # Check if function is imported
        import_analysis = self._analyze_import_usage(func_name)
        
        # Check if function is exported
        export_analysis = self._analyze_export_usage(func_name)
        
        # Determine validation status
        validation_status = self._determine_validation_status(
            usage_analysis, import_analysis, export_analysis
        )
        
        return {
            'function_name': func_name,
            'file_path': func_def.file_path,
            'line_number': func_def.line_number,
            'validation_status': validation_status,
            'usage_analysis': usage_analysis,
            'import_analysis': import_analysis,
            'export_analysis': export_analysis,
            'is_truly_unused': validation_status == 'truly_unused',
            'risk_level': self._assess_removal_risk(validation_status, usage_analysis)
        }
    
    def _analyze_usage_patterns(self, usage_patterns: List['UsageInfo']) -> Dict[str, Any]:
        """Analyze usage patterns for a function."""
        if not usage_patterns:
            return {
                'total_usage': 0,
                'usage_types': {},
                'files_using': [],
                'has_meaningful_usage': False
            }
        
        usage_types = Counter(usage.usage_type for usage in usage_patterns)
        files_using = list({usage.file_path for usage in usage_patterns})
        
        # Determine if usage is meaningful
        meaningful_types = {'function_call', 'method_call', 'attribute_access'}
        has_meaningful_usage = any(usage_type in meaningful_types for usage_type in usage_types)
        
        return {
            'total_usage': len(usage_patterns),
            'usage_types': dict(usage_types),
            'files_using': files_using,
            'has_meaningful_usage': has_meaningful_usage
        }
    
    def _analyze_import_usage(self, func_name: str) -> Dict[str, Any]:
        """Analyze import usage of a function."""
        import_usage = []
        
        for file_path, imports in self.import_statements.items():
            for import_info in imports:
                if func_name in import_info.module_name or func_name == import_info.alias_name:
                    import_usage.append({
                        'file_path': file_path,
                        'module_name': import_info.module_name,
                        'alias_name': import_info.alias_name,
                        'line_number': import_info.line_number,
                        'import_type': import_info.import_type
                    })
        
        return {
            'is_imported': len(import_usage) > 0,
            'import_count': len(import_usage),
            'import_details': import_usage
        }
    
    def _analyze_export_usage(self, func_name: str) -> Dict[str, Any]:
        """Analyze export usage of a function."""
        export_usage = []
        
        for file_path, exports in self.export_statements.items():
            for export_info in exports:
                if func_name == export_info.name:
                    export_usage.append({
                        'file_path': file_path,
                        'name': export_info.name,
                        'line_number': export_info.line_number,
                        'export_type': export_info.export_type
                    })
        
        return {
            'is_exported': len(export_usage) > 0,
            'export_count': len(export_usage),
            'export_details': export_usage
        }
    
    def _determine_validation_status(self, usage_analysis: Dict, import_analysis: Dict, export_analysis: Dict) -> str:
        """Determine the validation status of a function."""
        if export_analysis['is_exported']:
            return 'exported_function'
        elif import_analysis['is_imported']:
            return 'imported_function'
        elif usage_analysis['has_meaningful_usage']:
            return 'used_function'
        elif usage_analysis['total_usage'] == 0:
            return 'truly_unused'
        else:
            return 'potentially_unused'
    
    def _assess_removal_risk(self, validation_status: str, usage_analysis: Dict) -> str:
        """Assess the risk level of removing a function."""
        if validation_status == 'exported_function':
            return 'critical'
        elif validation_status == 'imported_function':
            return 'high'
        elif validation_status == 'used_function':
            return 'medium'
        elif validation_status == 'truly_unused':
            return 'low'
        else:
            return 'unknown'
    
    def get_truly_unused_functions(self) -> List[Dict[str, Any]]:
        """Get list of functions that are truly unused."""
        truly_unused = []
        
        for func_name, validation_result in self.validation_results.items():
            if validation_result['is_truly_unused']:
                truly_unused.append(validation_result)
        
        return sorted(truly_unused, key=lambda x: x['file_path'])
    
    def get_high_risk_functions(self) -> List[Dict[str, Any]]:
        """Get list of functions that are high risk to remove."""
        high_risk = []
        
        for func_name, validation_result in self.validation_results.items():
            if validation_result['risk_level'] in ['high', 'critical']:
                high_risk.append(validation_result)
        
        return sorted(high_risk, key=lambda x: x['file_path'])
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "summary": {
                "total_functions": len(self.function_definitions),
                "truly_unused": len([r for r in self.validation_results.values() if r['is_truly_unused']]),
                "used_functions": len([r for r in self.validation_results.values() if not r['is_truly_unused']]),
                "files_with_syntax_errors": len(self.syntax_errors),
                "validation_complete": True
            },
            "validation_results": self.validation_results,
            "truly_unused_functions": self.get_truly_unused_functions(),
            "high_risk_functions": self.get_high_risk_functions(),
            "syntax_errors": {
                str(k): v for k, v in self.syntax_errors.items()
            }
        }
        
        return report
    
    def save_report(self, output_path: str = "function_usage_validation.json"):
        """Save validation report as JSON."""
        report = self.generate_validation_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Function usage validation report saved to {output_path}")
    
    def print_summary(self):
        """Print validation summary."""
        report = self.generate_validation_report()
        summary = report["summary"]
        
        print(f"\n{'='*80}")
        print(f"FUNCTION USAGE VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"📁 Total functions analyzed: {summary['total_functions']}")
        print(f"✅ Truly unused functions: {summary['truly_unused']}")
        print(f"🔗 Used functions: {summary['used_functions']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        
        truly_unused = self.get_truly_unused_functions()
        if truly_unused:
            print(f"\n🗑️  TRULY UNUSED FUNCTIONS (Top 20):")
            for i, func_info in enumerate(truly_unused[:20], 1):
                print(f"   {i:2d}. {func_info['function_name']} in {func_info['file_path']}")
        
        high_risk = self.get_high_risk_functions()
        if high_risk:
            print(f"\n🚨 HIGH RISK FUNCTIONS (Top 10):")
            for i, func_info in enumerate(high_risk[:10], 1):
                print(f"   {i:2d}. {func_info['function_name']} in {func_info['file_path']}")
                print(f"       Risk: {func_info['risk_level']}")

# Data classes for structured information
class FunctionDefinition:
    def __init__(self, name: str, file_path: str, line_number: int, **kwargs):
        self.name = name
        self.file_path = file_path
        self.line_number = line_number
        self.is_class_method = kwargs.get('is_class_method', False)
        self.class_name = kwargs.get('class_name')
        self.decorators = kwargs.get('decorators', [])
        self.arguments = kwargs.get('arguments', [])
        self.has_docstring = kwargs.get('has_docstring', False)
        self.is_async = kwargs.get('is_async', False)
        self.body_lines = kwargs.get('body_lines', 0)
        self.extracted_with_regex = kwargs.get('extracted_with_regex', False)

class UsageInfo:
    def __init__(self, function_name: str, file_path: str, usage_type: str, line_number: int, context: str):
        self.function_name = function_name
        self.file_path = file_path
        self.usage_type = usage_type
        self.line_number = line_number
        self.context = context

class ImportInfo:
    def __init__(self, module_name: str, alias_name: str, file_path: str, line_number: int, import_type: str):
        self.module_name = module_name
        self.alias_name = alias_name
        self.file_path = file_path
        self.line_number = line_number
        self.import_type = import_type

class ExportInfo:
    def __init__(self, name: str, file_path: str, line_number: int, export_type: str):
        self.name = name
        self.file_path = file_path
        self.line_number = line_number
        self.export_type = export_type

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Function usage validation for: {root_dir}")
    
    validator = FunctionUsageValidator(root_dir)
    
    # Validate all functions
    validator.validate_all_functions()
    
    # Generate outputs
    validator.save_report()
    
    # Print summary
    validator.print_summary()

if __name__ == "__main__":
    main()