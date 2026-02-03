
import re

def remove_orphan_markers(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        if line.startswith('>>>>>>> origin/codex/analyze-process-improvement-for-stuck-execution'):
            continue
        clean_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(clean_lines)
    print("Cleaned orphan markers")

remove_orphan_markers('src/training/steps/labeling/label_based_layer_2.py')
