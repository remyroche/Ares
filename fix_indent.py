
def replace_method():
    with open('src/training/steps/labeling/label_based_layer_2.py', 'r') as f:
        content = f.read()

    # 1. Fix unexpected indent in LAYER2_MODEL_CONSTANTS
    # The previous sed output showed:
    # # Constants for Layer 2 Model Training (defaults/fixed) - HIGH REGULARIZATION for noisy financial dataLAYER2_MODEL_CONSTANTS = {
    # This means the comment and variable declaration merged or indentation is weird.

    # Let's fix it by splitting the line
    content = content.replace(
        '# Constants for Layer 2 Model Training (defaults/fixed) - HIGH REGULARIZATION for noisy financial dataLAYER2_MODEL_CONSTANTS = {',
        '# Constants for Layer 2 Model Training (defaults/fixed) - HIGH REGULARIZATION for noisy financial data\nLAYER2_MODEL_CONSTANTS = {'
    )

    # Or cleaner:
    content = content.replace(
        '# Constants for Layer 2 Model Training (defaults/fixed) - EXTREME REGULARIZATION for Path StabilityLAYER2_MODEL_CONSTANTS = {',
        '# Constants for Layer 2 Model Training (defaults/fixed) - EXTREME REGULARIZATION for Path Stability\nLAYER2_MODEL_CONSTANTS = {'
    )

    # Also fix potential issue where there is no newline before LAYER2_MODEL_CONSTANTS
    # Use regex to be safe
    import re
    content = re.sub(r'(#.*?)LAYER2_MODEL_CONSTANTS', r'\1\nLAYER2_MODEL_CONSTANTS', content)

    with open('src/training/steps/labeling/label_based_layer_2.py', 'w') as f:
        f.write(content)

    print("Applied indent fix successfully")

replace_method()
