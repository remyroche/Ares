
import re

def resolve_conflict(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Debug
    print(f"Content length: {len(content)}")
    if '<<<<<<< HEAD' in content:
        print("Found <<<<<<< HEAD literal")
    else:
        print("Did NOT find <<<<<<< HEAD literal")

    # Relaxed regex
    # Match <<<<<<< HEAD, then anything (lazy) until =======, then anything (lazy) until >>>>>>> ...
    # We use re.DOTALL so . matches newlines

    # We also handle potential whitespace around markers
    pattern = re.compile(r'<<<<<<< HEAD\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> .*?\n', re.DOTALL)

    def replacement(match):
        head_block = match.group(1)
        incoming_block = match.group(2)

        # 1. Imports: Check for _numba_rolling_std
        if '_numba_rolling_std' in head_block and '_numba_rolling_kurt' in incoming_block:
            return incoming_block + "\n"

        # 2. Constants: Check for LAYER2_MODEL_CONSTANTS
        if 'LAYER2_MODEL_CONSTANTS =' in head_block:
            return head_block + "\n" # Add newline back if we stripped it

        # 3. Constants: Check for LAYER2_PROBE_CONSTANTS
        if 'LAYER2_PROBE_CONSTANTS =' in head_block:
            return head_block + "\n"

        # 4. Feature computation method (optimization)
        if 'def _compute_specific_geometry_features' in head_block or 'def _compute_specific_geometry_features' in incoming_block:
            return incoming_block + "\n"

        # 5. Logic inside _compute_specific_geometry_features
        if 'tprint_info' in incoming_block and 'tprint_info' not in head_block:
             return incoming_block + "\n"

        if 'apply_rolling' in incoming_block and 'label=' in incoming_block:
             return incoming_block + "\n"

        if 'fam_cache_key' in incoming_block:
             return incoming_block + "\n"

        if '_tmp_row_id' in incoming_block:
             return incoming_block + "\n"

        return incoming_block + "\n"

    resolved_content, count = pattern.subn(replacement, content)
    print(f"Replaced {count} occurrences")

    with open(file_path, 'w') as f:
        f.write(resolved_content)

resolve_conflict('src/training/steps/labeling/label_based_layer_2.py')
