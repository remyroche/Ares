import re

# Read the file
with open('label_based_layer_2.py', 'r') as f:
    content = f.read()

# Add Bayesian discovery configuration
config_pattern = r'(        # Causal Discovery Parameters\s+self\.significance_level = kwargs\.get\("significance_level", 0\.05)\s+self\.max_conditioning_set = kwargs\.get\("max_conditioning_set", 3)\s+self\.use_lingam = kwargs\.get\("use_lingam", True))'

config_replacement = r'''        # Causal Discovery Parameters
        self.significance_level = kwargs.get("significance_level", 0.05)
        self.max_conditioning_set = kwargs.get("max_conditioning_set", 3)
        self.use_lingam = kwargs.get("use_lingam", True)
        self.use_bayesian_discovery = kwargs.get("use_bayesian_discovery", True)
        self.bayesian_n_bootstrap = kwargs.get("bayesian_n_bootstrap", 50)'''

# Apply the replacement
content = re.sub(config_pattern, config_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back to file
with open('label_based_layer_2.py', 'w') as f:
    f.write(content)

print("Added Bayesian discovery configuration to label_based_layer_2.py")
