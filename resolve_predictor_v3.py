
import re

def resolve_predictor(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex for conflict blocks - allow optional whitespace
    # We want to keep HEAD content only
    pattern = re.compile(r'<<<<<<< HEAD\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> .*?\n', re.DOTALL)

    def replacement(match):
        head_block = match.group(1)
        # Add newline if not present at end of head block to maintain spacing?
        # The regex captured up to ======= so it might include trailing newline or not depending on spacing
        # Let's ensure proper spacing.
        return head_block + "\n"

    resolved_content, count = pattern.subn(replacement, content)
    print(f"Replaced {count} occurrences")

    with open(file_path, 'w') as f:
        f.write(resolved_content)

resolve_predictor('src/training/steps/labeling/predictor_geometry_generators.py')
