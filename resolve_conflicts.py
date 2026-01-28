import re
import sys

def resolve_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Regex to find conflict blocks
    # <<<<<<< HEAD
    # ... (ours)
    # =======
    # ... (theirs)
    # >>>>>>> origin/...
    
    pattern = re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> .*?\n', re.DOTALL)
    
    def replacer(match):
        # We want "theirs" (group 2)
        return match.group(2) + '\n'
        
    new_content = pattern.sub(replacer, content)
    
    # Check if any conflicts remain (e.g. nested or different markers)
    if '<<<<<<<' in new_content:
        print(f"Warning: Potential remaining conflicts in {filepath}")
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Resolved conflicts in {filepath}")

resolve_file('src/training/steps/labeling/label_based_layer_2.py')
