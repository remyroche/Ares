#!/usr/bin/env python3
"""
Fix syntax errors in problem_specific_strategies.py
"""

def fix_syntax_errors():
    """Fix all syntax errors in the file."""
    with open('src/training/optimization/problem_specific_strategies.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix function parameter syntax errors
    content = content.replace('def _detect_noise(self = values: np.ndarray) -> bool:', 
                             'def _detect_noise(self, values: np.ndarray) -> bool:')
    
    # Fix function parameter syntax errors
    content = content.replace('def _detect_multi_modality(self, points: np.ndarray = values: np.ndarray) -> bool:', 
                             'def _detect_multi_modality(self, points: np.ndarray, values: np.ndarray) -> bool:')
    
    # Fix function parameter syntax errors
    content = content.replace('def _detect_multi_objective(self = values: np.ndarray) -> bool:', 
                             'def _detect_multi_objective(self, values: np.ndarray) -> bool:')
    
    # Fix list syntax errors
    content = content.replace('constraint_indicators = [\'constraint\' = \'bound\', \'limit\', \'range\']', 
                             'constraint_indicators = [\'constraint\', \'bound\', \'limit\', \'range\']')
    
    # Fix isinstance syntax errors
    content = content.replace('if isinstance(param_config = dict):', 
                             'if isinstance(param_config, dict):')
    
    # Fix comment syntax errors
    content = content.replace('# If there are many small differences = it might be noisy', 
                             '# If there are many small differences, it might be noisy')
    
    with open('src/training/optimization/problem_specific_strategies.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    fix_syntax_errors()
    print("Fixed syntax errors in problem_specific_strategies.py")