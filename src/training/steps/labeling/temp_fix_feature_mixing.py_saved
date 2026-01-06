import re

# Read the file
with open('causal_feature_engineering.py', 'r') as f:
    content = f.read()

# Add replace_features parameter to __init__
pattern = r'(        self.verbose = verbose\n\n        # Initialize DML models)'

replacement = r'''        self.verbose = verbose
        
        # Configuration for feature mixing behavior
        self.replace_features = kwargs.get('replace_features', False)
        self.feature_suffix = kwargs.get('feature_suffix', '_causal_adjusted')
        
        # Initialize DML models'''

# Apply the replacement
new_content = re.sub(pattern, replacement, content)

# Write back to file
with open('causal_feature_engineering.py', 'w') as f:
    f.write(new_content)

print("Added feature mixing configuration")
