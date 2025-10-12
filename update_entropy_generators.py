#!/usr/bin/env python3
"""
Script to update entropy generators to use VectorBT optimization.
This script will systematically update all entropy generators to inherit from BaseEntropyGenerator
and remove duplicate methods.
"""

import re

def update_entropy_file():
    """Update the entropy.py file to use VectorBT optimization."""
    
    # Read the current file
    with open('/workspace/src/feature_generation/categories/entropy.py', 'r') as f:
        content = f.read()
    
    # Pattern to find generator classes that need updating
    generator_pattern = r'class (\w+EntropyGenerator)\(VectorizedFeatureGenerator\):'
    
    # Find all generator classes
    generators = re.findall(generator_pattern, content)
    print(f"Found {len(generators)} entropy generators to update:")
    for gen in generators:
        print(f"  - {gen}")
    
    # Update each generator class
    for generator in generators:
        if generator in ['EntropyFeatureGenerator', 'PriceEntropyGenerator', 'VolumeEntropyGenerator']:
            continue  # Already updated
            
        # Update class inheritance
        old_class_def = f'class {generator}(VectorizedFeatureGenerator):'
        new_class_def = f'class {generator}(BaseEntropyGenerator):'
        content = content.replace(old_class_def, new_class_def)
        
        # Update class docstring
        class_pattern = rf'class {generator}\(BaseEntropyGenerator\):\s*\n\s*"""[^"]*"""'
        class_match = re.search(class_pattern, content)
        if class_match:
            old_docstring = class_match.group(0)
            new_docstring = old_docstring.replace('Generator for', 'Generator for').replace('Generator for', 'Generator for with VectorBT optimization')
            content = content.replace(old_docstring, new_docstring)
    
    # Remove duplicate optimize_dataframe_processing and vectorized_rolling_operations methods
    # These are now inherited from BaseEntropyGenerator
    
    # Pattern to find and remove duplicate methods
    duplicate_method_pattern = r'    def optimize_dataframe_processing\(self, data: pd\.DataFrame\) -> pd\.DataFrame:\s*\n\s*"""Optimize DataFrame for vectorized processing\."""\s*\n\s*if hasattr\(self, \'vectorization_optimizer\'\) and self\.vectorization_optimizer:\s*\n\s*return self\.vectorization_optimizer\.optimize_dataframe_processing\(data\)\s*\n\s*return data\s*\n\s*def vectorized_rolling_operations\(self, data: pd\.DataFrame, operations: List\[str\], \s*\n\s*windows: List\[int\], columns: Optional\[List\[str\]\] = None\) -> pd\.DataFrame:\s*\n\s*"""Perform vectorized rolling operations with hardware optimization\."""\s*\n\s*if hasattr\(self, \'vectorization_optimizer\'\) and self\.vectorization_optimizer:\s*\n\s*return self\.vectorization_optimizer\.vectorized_rolling_operations\(\s*\n\s*data, operations, windows, columns\s*\n\s*\)\s*\n\s*return data'
    
    # Remove all duplicate methods
    content = re.sub(duplicate_method_pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Update _generate_feature methods to use VectorBT optimization
    def update_generate_feature(match):
        class_name = match.group(1)
        method_content = match.group(2)
        
        # Add VectorBT optimization to the method
        if 'calculate_vectorized_entropy' in method_content:
            # Update existing calculate_vectorized_entropy calls
            method_content = re.sub(
                r'calculate_vectorized_entropy\(([^,]+), ([^)]+)\)',
                r'calculate_vectorized_entropy(\1, \2, use_vectorbt=self.use_vectorbt)',
                method_content
            )
        
        # Add data optimization
        if 'optimize_dataframe_processing' not in method_content:
            method_content = method_content.replace(
                'def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:',
                'def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:\n        # Optimize DataFrame for processing\n        data = self.optimize_dataframe_processing(data)'
            )
        
        return f'class {class_name}(BaseEntropyGenerator):\n{method_content}'
    
    # Pattern to find _generate_feature methods
    generate_feature_pattern = r'class (\w+EntropyGenerator)\(BaseEntropyGenerator\):.*?def _generate_feature\(self, data: pd\.DataFrame, \*\*kwargs\) -> pd\.Series:.*?(?=class|\Z)'
    
    content = re.sub(generate_feature_pattern, update_generate_feature, content, flags=re.MULTILINE | re.DOTALL)
    
    # Write the updated content back
    with open('/workspace/src/feature_generation/categories/entropy.py', 'w') as f:
        f.write(content)
    
    print("✅ Successfully updated entropy generators with VectorBT optimization!")

if __name__ == "__main__":
    update_entropy_file()