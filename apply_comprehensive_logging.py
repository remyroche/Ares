#!/usr/bin/env python3
"""
Script to systematically apply comprehensive logging to all training step functions.

This script will:
1. Find all step files in src/training/steps/
2. Identify functions that need logging
3. Apply appropriate logging decorators
4. Ensure consistent logging patterns across all steps
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild('LoggingEnhancer')

class FunctionAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python functions."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.current_class = None
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        self.current_class = node.name
        self.classes.append({
            'name': node.name,
            'line': node.lineno,
            'methods': []
        })
        self.generic_visit(node)
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        func_info = {
            'name': node.name,
            'line': node.lineno,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'is_method': self.current_class is not None,
            'class_name': self.current_class,
            'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list]
        }
        
        if self.current_class:
            # Add to current class methods
            for cls in self.classes:
                if cls['name'] == self.current_class:
                    cls['methods'].append(func_info)
                    break
        else:
            # Standalone function
            self.functions.append(func_info)
        
        self.generic_visit(node)
    
    def _get_decorator_name(self, decorator):
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.value.id}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return f"{decorator.func.value.id}.{decorator.func.attr}"
        return "unknown"

class LoggingEnhancer:
    """Enhances Python files with comprehensive logging."""
    
    def __init__(self):
        self.steps_dir = project_root / "src" / "training" / "steps"
        self.enhanced_files = []
        self.skipped_files = []
        self.error_files = []
        
        # Functions that should have comprehensive logging
        self.target_functions = {
            'execute', 'execute_logic', 'initialize', '_initialize_step',
            'validate_inputs', 'validate_data', 'process_data', 'train_model',
            'predict', 'evaluate', 'save_model', 'load_model', 'optimize',
            'detect_regimes', 'engineer_features', 'select_features',
            'split_data', 'label_data', 'backtest', 'validate_results'
        }
        
        # Functions that should have important call logging
        self.important_functions = {
            '__init__', 'setup', 'configure', 'prepare', 'cleanup',
            'preprocess', 'postprocess', 'transform', 'fit', 'score'
        }
        
        # Files to skip (already have good logging or are utilities)
        self.skip_patterns = [
            '__pycache__',
            '.pyc',
            'test_',
            '_test.py',
            'validator.py',
            'utils.py',
            'decorators.py',
            'config.py'
        ]
    
    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        file_str = str(file_path)
        return any(pattern in file_str for pattern in self.skip_patterns)
    
    def find_step_files(self) -> List[Path]:
        """Find all step files that need logging enhancement."""
        step_files = []
        
        for root, dirs, files in os.walk(self.steps_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not self.should_skip_file(file_path):
                        step_files.append(file_path)
        
        logger.info(f"Found {len(step_files)} step files to analyze")
        return step_files
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a Python file to understand its structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            analyzer = FunctionAnalyzer()
            analyzer.visit(tree)
            
            return {
                'path': file_path,
                'content': content,
                'functions': analyzer.functions,
                'classes': analyzer.classes,
                'lines': content.split('\n')
            }
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def needs_logging_enhancement(self, file_info: Dict) -> bool:
        """Check if file needs logging enhancement."""
        if not file_info:
            return False
        
        # Check if file already imports comprehensive logging
        content = file_info['content']
        if 'comprehensive_function_logger' in content:
            return False
        
        # Check if file has any target functions
        all_functions = file_info['functions'] + []
        for cls in file_info['classes']:
            all_functions.extend(cls['methods'])
        
        target_funcs = [f for f in all_functions if f['name'] in self.target_functions]
        important_funcs = [f for f in all_functions if f['name'] in self.important_functions]
        
        return len(target_funcs) > 0 or len(important_funcs) > 0
    
    def enhance_file(self, file_info: Dict) -> bool:
        """Enhance a file with comprehensive logging."""
        try:
            file_path = file_info['path']
            content = file_info['content']
            lines = file_info['lines']
            
            # Add import for comprehensive logging
            import_line = "from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation\n"
            
            # Find the best place to insert the import
            import_inserted = False
            new_lines = []
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # Insert import after other imports
                if (line.startswith('import ') or line.startswith('from ')) and not import_inserted:
                    # Check if next line is also an import
                    if i + 1 < len(lines) and (lines[i + 1].startswith('import ') or lines[i + 1].startswith('from ') or lines[i + 1].strip() == ''):
                        continue
                    else:
                        # Insert our import here
                        new_lines.append(import_line)
                        import_inserted = True
            
            if not import_inserted:
                # Insert at the beginning after docstring
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    if line.strip() and not line.startswith('"""') and not line.startswith("'''") and not line.startswith('#'):
                        new_lines.insert(i, import_line)
                        import_inserted = True
                        break
            
            # Apply decorators to functions
            enhanced_lines = self._apply_decorators(new_lines, file_info)
            
            # Write enhanced file
            enhanced_content = '\n'.join(enhanced_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            
            logger.info(f"Enhanced {file_path} with comprehensive logging")
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing {file_path}: {e}")
            return False
    
    def _apply_decorators(self, lines: List[str], file_info: Dict) -> List[str]:
        """Apply logging decorators to appropriate functions."""
        enhanced_lines = lines.copy()
        
        # Track line number adjustments due to insertions
        line_offset = 0
        
        # Process all functions (standalone and methods)
        all_functions = file_info['functions'] + []
        for cls in file_info['classes']:
            all_functions.extend(cls['methods'])
        
        for func in all_functions:
            func_line = func['line'] + line_offset - 1  # Convert to 0-based index
            
            # Determine appropriate decorator
            decorator = None
            if func['name'] in self.target_functions:
                decorator = "@log_step_functions"
            elif func['name'] in self.important_functions:
                decorator = "@log_important_calls"
            elif func['name'].startswith('_') and func['is_method']:
                decorator = "@log_all_calls"
            
            if decorator and decorator not in func['decorators']:
                # Insert decorator before function definition
                enhanced_lines.insert(func_line, f"    {decorator}" if func['is_method'] else decorator)
                line_offset += 1
        
        return enhanced_lines
    
    def run(self):
        """Run the logging enhancement process."""
        logger.info("Starting comprehensive logging enhancement for training steps")
        
        # Find all step files
        step_files = self.find_step_files()
        
        # Analyze and enhance files
        for file_path in step_files:
            logger.info(f"Analyzing {file_path}")
            
            file_info = self.analyze_file(file_path)
            if not file_info:
                self.error_files.append(file_path)
                continue
            
            if self.needs_logging_enhancement(file_info):
                if self.enhance_file(file_info):
                    self.enhanced_files.append(file_path)
                else:
                    self.error_files.append(file_path)
            else:
                self.skipped_files.append(file_path)
        
        # Report results
        logger.info(f"Logging enhancement completed:")
        logger.info(f"  Enhanced files: {len(self.enhanced_files)}")
        logger.info(f"  Skipped files: {len(self.skipped_files)}")
        logger.info(f"  Error files: {len(self.error_files)}")
        
        if self.enhanced_files:
            logger.info("Enhanced files:")
            for file_path in self.enhanced_files:
                logger.info(f"  - {file_path}")
        
        if self.error_files:
            logger.error("Files with errors:")
            for file_path in self.error_files:
                logger.error(f"  - {file_path}")

def main():
    """Main function."""
    enhancer = LoggingEnhancer()
    enhancer.run()

if __name__ == "__main__":
    main()

