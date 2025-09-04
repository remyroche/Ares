#!/usr/bin/env python3
"""
Debug script to understand the undefined names analyzer behavior.
"""

import ast
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_quality.analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer

def test_analyzer_on_file(file_path: str):
    """Test the analyzer on a specific file."""
    print(f"\n{'='*60}")
    print(f"Testing analyzer on: {file_path}")
    print(f"{'='*60}")
    
    analyzer = UndefinedNamesAnalyzer()
    result = analyzer.analyze_file(file_path)
    
    print(f"Status: {result['status']}")
    print(f"Total errors: {result['total_errors']}")
    
    if result['total_errors'] > 0:
        print("\nErrors found:")
        for error in result['errors']:
            print(f"  Line {error['line']}: {error['name']} - {error['context']}")
    
    return result

def test_specific_function():
    """Test the analyzer's parameter recognition logic."""
    print(f"\n{'='*60}")
    print("Testing parameter recognition logic")
    print(f"{'='*60}")
    
    # Test code
    test_code = '''
async def validate_step_outputs(self, symbol: str, exchange: str, data_dir: str):
    """Validate Step 9.5 outputs."""
    try:
        self.logger.info(f"🔍 Validating Step 9.5 outputs for {symbol} on {exchange}")
        return True
    except Exception as e:
        return False
'''
    
    # Parse AST
    tree = ast.parse(test_code)
    
    # Find the function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'validate_step_outputs':
            func_node = node
            break
    
    if func_node:
        print(f"Function: {func_node.name}")
        print(f"Parameters: {[arg.arg for arg in func_node.args.args]}")
        
        # Find symbol usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == 'symbol' and isinstance(node.ctx, ast.Load):
                print(f"\nSymbol usage found at line {node.lineno}")
                
                # Test the analyzer's logic
                analyzer = UndefinedNamesAnalyzer()
                
                # Check if it's in function context
                is_in_context = analyzer._is_in_function_context(node, tree)
                print(f"Is in function context: {is_in_context}")
                
                # Check if it's in function body
                is_in_body = analyzer._is_node_in_function_body(node, func_node)
                print(f"Is in function body: {is_in_body}")

if __name__ == "__main__":
    # Test on the problematic file
    test_analyzer_on_file("src/training/steps/model_training/step09_5_multi_timeframe_hmm_ensemble_validator.py")
    
    # Test the specific function logic
    test_specific_function()
