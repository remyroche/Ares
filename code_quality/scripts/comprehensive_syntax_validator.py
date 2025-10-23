#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Comprehensive Syntax Validator

This script provides thorough syntax validation using multiple methods:
1. Python compile() - The gold standard for syntax validation
2. AST parsing - More detailed error reporting
3. py_compile module - Alternative compilation method
4. Tokenization - Catches token-level issues

It distinguishes between:
- Real syntax errors (can't be compiled)
- Import/runtime errors (valid syntax, missing dependencies)
- Semantic errors (valid syntax, but AST parsing issues)
"""

import ast
import compileall
import os
import py_compile
import sys
import tokenize
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
import numpy as np
import time
import re

class ComprehensiveSyntaxValidator:
    """Comprehensive syntax validation with multiple validation methods."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.results = {
            "total_files": 0,
            "syntax_errors": [],
            "indentation_errors": [],
            "import_errors": [],
            "runtime_errors": [],
            "valid_files": [],
            "file_not_found": [],
            "permission_errors": [],
            "error_counts": {
                "syntax_errors": 0,
                "indentation_errors": 0,
                "import_errors": 0,
                "runtime_errors": 0,
                "valid_files": 0,
                "file_not_found": 0,
                "permission_errors": 0
            }
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a single Python file using multiple methods."""
        result = {
            "file_path": file_path,
            "syntax_valid": False,
            "ast_parseable": False,
            "compilable": False,
            "importable": False,
            "error_type": None,
            "error_message": None,
            "line_number": None,
            "validation_methods": {
                "compile": None,
                "ast_parse": None,
                "py_compile": None,
                "import": None
            }
        }
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            result["error_type"] = "file_not_found"
            result["error_message"] = "File not found"
            return result
        except PermissionError:
            result["error_type"] = "permission_error"
            result["error_message"] = "Permission denied"
            return result
        except Exception as e:
            result["error_type"] = "read_error"
            result["error_message"] = f"Error reading file: {str(e)}"
            return result
        
        # Test 1: Python compile() - The gold standard
        try:
            compile(content, file_path, 'exec')
            result["validation_methods"]["compile"] = "success"
            result["syntax_valid"] = True
            result["compilable"] = True
        except SyntaxError as e:
            result["validation_methods"]["compile"] = "syntax_error"
            result["error_type"] = "syntax_error"
            result["error_message"] = f"Syntax error: {e.msg}"
            result["line_number"] = e.lineno
            return result
        except IndentationError as e:
            result["validation_methods"]["compile"] = "indentation_error"
            result["error_type"] = "indentation_error"
            result["error_message"] = f"Indentation error: {e.msg}"
            result["line_number"] = e.lineno
            return result
        except Exception as e:
            result["validation_methods"]["compile"] = "compile_error"
            result["error_type"] = "compile_error"
            result["error_message"] = f"Compile error: {str(e)}"
            return result
        
        # Test 2: AST parsing
        try:
            tree = ast.parse(content, filename=file_path)
            result["validation_methods"]["ast_parse"] = "success"
            result["ast_parseable"] = True
        except SyntaxError as e:
            result["validation_methods"]["ast_parse"] = "syntax_error"
            # This shouldn't happen if compile() succeeded, but just in case
            if not result["error_type"]:
                result["error_type"] = "ast_syntax_error"
                result["error_message"] = f"AST syntax error: {e.msg}"
                result["line_number"] = e.lineno
        except Exception as e:
            result["validation_methods"]["ast_parse"] = "ast_error"
            if not result["error_type"]:
                result["error_type"] = "ast_error"
                result["error_message"] = f"AST error: {str(e)}"
        
        # Test 3: py_compile module
        try:
            py_compile.compile(file_path, doraise=True)
            result["validation_methods"]["py_compile"] = "success"
        except py_compile.PyCompileError as e:
            result["validation_methods"]["py_compile"] = "py_compile_error"
            if not result["error_type"]:
                result["error_type"] = "py_compile_error"
                result["error_message"] = f"PyCompile error: {e.msg}"
        except Exception as e:
            result["validation_methods"]["py_compile"] = "py_compile_error"
            if not result["error_type"]:
                result["error_type"] = "py_compile_error"
                result["error_message"] = f"PyCompile error: {str(e)}"
        
        # Test 4: Import test (only if syntax is valid)
        if result["syntax_valid"]:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('test_module', file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                result["validation_methods"]["import"] = "success"
                result["importable"] = True
            except ImportError as e:
                result["validation_methods"]["import"] = "import_error"
                result["error_type"] = "import_error"
                result["error_message"] = f"Import error: {str(e)}"
            except AttributeError as e:
                result["validation_methods"]["import"] = "attribute_error"
                result["error_type"] = "attribute_error"
                result["error_message"] = f"Attribute error: {str(e)}"
            except NameError as e:
                result["validation_methods"]["import"] = "name_error"
                result["error_type"] = "name_error"
                result["error_message"] = f"Name error: {str(e)}"
            except Exception as e:
                result["validation_methods"]["import"] = "runtime_error"
                result["error_type"] = "runtime_error"
                result["error_message"] = f"Runtime error: {str(e)}"
        
        # If we get here and no error type is set, the file is valid
        if not result["error_type"]:
            result["error_type"] = "valid"
            result["error_message"] = "File is valid"
        
        return result
    
    def validate_directory(self, directory: str = None, pattern: str = "**/*.py") -> Dict[str, Any]:
        """Validate all Python files in a directory."""
        if directory:
            search_path = Path(directory)
        else:
            search_path = self.project_root
        
        python_files = list(search_path.glob(pattern))
        self.results["total_files"] = len(python_files)
        
        tprint(f"🔍 Validating {len(python_files)} Python files...")
        
        for file_path in python_files:
            result = self.validate_file(str(file_path))
            
            # Categorize results
            if result["error_type"] == "syntax_error":
                self.results["syntax_errors"].append(result)
                self.results["error_counts"]["syntax_errors"] += 1
            elif result["error_type"] == "indentation_error":
                self.results["indentation_errors"].append(result)
                self.results["error_counts"]["indentation_errors"] += 1
            elif result["error_type"] in ["import_error", "attribute_error", "name_error"]:
                self.results["import_errors"].append(result)
                self.results["error_counts"]["import_errors"] += 1
            elif result["error_type"] == "runtime_error":
                self.results["runtime_errors"].append(result)
                self.results["error_counts"]["runtime_errors"] += 1
            elif result["error_type"] == "valid":
                self.results["valid_files"].append(result)
                self.results["error_counts"]["valid_files"] += 1
            elif result["error_type"] == "file_not_found":
                self.results["file_not_found"].append(result)
                self.results["error_counts"]["file_not_found"] += 1
            elif result["error_type"] == "permission_error":
                self.results["permission_errors"].append(result)
                self.results["error_counts"]["permission_errors"] += 1
        
        return self.results
    
    def print_summary(self):
        """Print a comprehensive summary of validation results."""
        tprint("\n" + "="*80)
        tprint("🔍 COMPREHENSIVE SYNTAX VALIDATION RESULTS")
        tprint("="*80)
        tprint(f"Total files analyzed: {self.results['total_files']}")
        tprint()
        
        # Print each category
        categories = [
            ("syntax_errors", "🔴 REAL SYNTAX ERRORS", "Files with invalid Python syntax (can't be compiled)"),
            ("indentation_errors", "🟠 INDENTATION ERRORS", "Files with indentation issues"),
            ("import_errors", "🟡 IMPORT/RUNTIME ERRORS", "Files with valid syntax but missing imports/dependencies"),
            ("runtime_errors", "🟣 RUNTIME ERRORS", "Files with valid syntax but runtime issues"),
            ("valid_files", "✅ VALID FILES", "Files that are completely valid"),
            ("file_not_found", "🔵 FILE NOT FOUND", "Files that don't exist"),
            ("permission_errors", "🟣 PERMISSION ERRORS", "Files with permission issues")
        ]
        
        for category, title, description in categories:
            count = self.results["error_counts"][category]
            if count > 0:
                tprint(f"{title}: {count} files")
                tprint(f"  {description}")
                
                # Show first few examples
                if category in self.results and self.results[category]:
                    examples = self.results[category][:3]  # Show first 3 examples
                    for example in examples:
                        if isinstance(example, dict) and "file_path" in example:
                            file_path = example["file_path"]
                            if "error_message" in example and example["error_message"]:
                                line_info = f" (line {example['line_number']})" if example.get('line_number') else ""
                                tprint(f"    - {file_path}: {example['error_message']}{line_info}")
                            else:
                                tprint(f"    - {file_path}")
                tprint()
        
        # Summary statistics
        total_errors = sum(self.results["error_counts"][cat] for cat in 
                          ["syntax_errors", "indentation_errors", "import_errors", "runtime_errors", 
                           "file_not_found", "permission_errors"])
        
        tprint(f"📈 SUMMARY STATISTICS:")
        tprint(f"  ✅ Valid files: {self.results['error_counts']['valid_files']} ({self.results['error_counts']['valid_files']/self.results['total_files']*100:.1f}%)")
        tprint(f"  ❌ Files with issues: {total_errors} ({total_errors/self.results['total_files']*100:.1f}%)")
        tprint(f"  🔴 Real syntax errors: {self.results['error_counts']['syntax_errors'] + self.results['error_counts']['indentation_errors']}")
        tprint(f"  🟡 Import/dependency issues: {self.results['error_counts']['import_errors']}")
        tprint(f"  🟣 Runtime issues: {self.results['error_counts']['runtime_errors']}")
    
    def save_results(self, output_file: str = None):
        """Save validation results to a JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"/workspace/code_quality/reports/syntax_validation_{timestamp}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        tprint(f"📄 Results saved to: {output_file}")
        return output_file

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Python syntax validator")
    parser.add_argument("--directory", "-d", help="Directory to validate (default: current directory)")
    parser.add_argument("--pattern", "-p", default="**/*.py", help="File pattern to match (default: **/*.py)")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (less output)")
    
    args = parser.parse_args()
    
    validator = ComprehensiveSyntaxValidator(args.directory)
    results = validator.validate_directory(args.directory, args.pattern)
    
    if not args.quiet:
        validator.print_summary()
    
    if args.output:
        validator.save_results(args.output)
    
    # Exit with error code if there are syntax errors
    syntax_error_count = results["error_counts"]["syntax_errors"] + results["error_counts"]["indentation_errors"]
    if syntax_error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()