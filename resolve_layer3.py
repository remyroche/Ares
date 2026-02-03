
import re

def resolve_conflict(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex for conflict blocks
    pattern = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)\n>>>>>>> .*?\n', re.DOTALL)

    def replacement(match):
        head_block = match.group(1)
        incoming_block = match.group(2)

        # 1. _generate_geometry_specific_nn_features definition
        if '_generate_geometry_specific_nn_features' in incoming_block and head_block.strip() == '':
            # Incoming adds this function. But we decided to reject if it's unused/disabled.
            # However, looking at the file content, it seems independent.
            # But generate_features_for_geometry (which uses it?) is disabled.
            # Let's reject it to keep clean.
            return head_block

        # 2. generate_features_for_geometry definition
        if 'def generate_features_for_geometry' in incoming_block and head_block.strip() == '':
            # Incoming adds this no-op function. Reject.
            return head_block

        # 3. logit_momentum_5 optimization
        if "features['logit_momentum_5']" in head_block and 'transform' in head_block:
            # HEAD uses transform (slow). Incoming uses reset_index (fast).
            # Accept incoming.
            return incoming_block + "\n"

        # 4. efficiency_ratio optimization
        if "features['efficiency_ratio']" in head_block and 'transform' in head_block:
            # Accept incoming.
            return incoming_block + "\n"

        # 5. y_alpha calculation
        if 'y_alpha =' in head_block:
            # HEAD has shift(-1). Incoming doesn't.
            # Keep HEAD.
            return head_block

        # 6. generate_features_for_geometry call
        if 'generate_features_for_geometry(alpha, gid)' in incoming_block:
            # Incoming adds call. Reject.
            return head_block

        # Default fallback: prefer HEAD if we don't recognize the conflict,
        # to be safe against overwriting logic I didn't analyze.
        # But wait, earlier for Layer 2 I defaulted to incoming for code fixes.
        # Here I analyzed most conflicts and want to reject incoming additions of disabled code.
        # But accept optimizations.

        # If I missed something, defaulting to HEAD is safer for logic preservation,
        # but defaulting to INCOMING is better for "merging new features".
        # Let's check if there are other conflicts.

        return head_block

    resolved_content, count = pattern.subn(replacement, content)
    print(f"Replaced {count} occurrences")

    with open(file_path, 'w') as f:
        f.write(resolved_content)

resolve_conflict('src/training/steps/labeling/label_based_layer_3_enhanced.py')
