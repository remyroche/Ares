import re

# Read the file
with open('interventionist_sampling.py', 'r') as f:
    content = f.read()

# Add new configuration parameters to __init__
pattern = r'(        self\.prediction_method = prediction_method\n\n        # Cache for structural shocks)'

replacement = r'''        self.prediction_method = prediction_method
        
        # Configuration for event mixing behavior
        self.separate_event_types = kwargs.get('separate_event_types', True)
        self.replace_cusum_events = kwargs.get('replace_cusum_events', False)
        self.min_event_distance = kwargs.get('min_event_distance', 5)
        
        # Cache for structural shocks'''

# Apply the replacement
new_content = re.sub(pattern, replacement, content)

# Write back to file
with open('interventionist_sampling.py', 'w') as f:
    f.write(new_content)

print("Added configuration options")
