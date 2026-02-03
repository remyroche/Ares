
def resolve_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    in_conflict = False
    head_lines = []
    incoming_lines = []
    current_marker = None # HEAD or =======

    for line in lines:
        if line.startswith('<<<<<<< HEAD'):
            in_conflict = True
            current_marker = 'HEAD'
            head_lines = []
            incoming_lines = []
            continue

        if in_conflict:
            if line.startswith('======='):
                current_marker = 'INCOMING'
                continue

            if line.startswith('>>>>>>> '):
                # End of conflict, resolve it
                resolved = resolve_block(head_lines, incoming_lines)
                output_lines.extend(resolved)
                in_conflict = False
                current_marker = None
                head_lines = []
                incoming_lines = []
                continue

            if current_marker == 'HEAD':
                head_lines.append(line)
            elif current_marker == 'INCOMING':
                incoming_lines.append(line)
        else:
            output_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(output_lines)

def resolve_block(head_lines, incoming_lines):
    head_text = ''.join(head_lines)
    incoming_text = ''.join(incoming_lines)

    # 1. Imports
    if '_numba_rolling_std' in head_text and '_numba_rolling_kurt' in incoming_text:
        return incoming_lines

    # 2. Constants: LAYER2_MODEL_CONSTANTS
    if 'LAYER2_MODEL_CONSTANTS =' in head_text:
        return head_lines # Keep HEAD

    # 3. Constants: LAYER2_PROBE_CONSTANTS
    if 'LAYER2_PROBE_CONSTANTS =' in head_text:
        return head_lines # Keep HEAD

    # 4. Feature computation (optimization)
    if 'def _compute_specific_geometry_features' in head_text or 'def _compute_specific_geometry_features' in incoming_text:
        return incoming_lines

    # 5. Logic inside feature computation
    if 'tprint_info' in incoming_text and 'tprint_info' not in head_text:
        return incoming_lines

    if 'apply_rolling' in incoming_text:
        return incoming_lines

    if 'fam_cache_key' in incoming_text:
        return incoming_lines

    if '_tmp_row_id' in incoming_text:
        return incoming_lines

    # Default to incoming for anything else
    return incoming_lines

resolve_file('src/training/steps/labeling/label_based_layer_2.py')
print("State machine resolution complete.")
