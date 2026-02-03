
def replace_method():
    with open('src/training/steps/labeling/label_based_layer_2.py', 'r') as f:
        content = f.read()

    with open('method_head.txt', 'r') as f:
        head_method = f.read()

    with open('method_codex.txt', 'r') as f:
        codex_method = f.read()

    # Imports
    content = content.replace(
        '_numba_rolling_std\n)',
        '_numba_rolling_std,\n    _numba_rolling_kurt\n)'
    )

    # Replace method
    # Use robust replacement since indentation might vary slightly
    # Find start index
    start_idx = content.find('    def _compute_specific_geometry_features(self, df, events_index, params):')
    if start_idx == -1:
        print("Could not find method start")
        return

    # Find end index (heuristic: look for next method or class end)
    # The method seems to be followed by _compute_rmi_scores in HEAD
    end_idx = content.find('    def _compute_rmi_scores', start_idx)

    if end_idx == -1:
        print("Could not find method end")
        return

    # Replace
    new_content = content[:start_idx] + codex_method + "\n\n" + content[end_idx:]

    # Clean up any leftover artifacts manually if needed
    # (The >>>>>> marker I saw earlier suggests manual editing error or incomplete resolution)
    # I should also remove that line 2596 marker.

    new_content = new_content.replace('>>>>>>> origin/codex/analyze-process-improvement-for-stuck-execution\n', '')

    with open('src/training/steps/labeling/label_based_layer_2.py', 'w') as f:
        f.write(new_content)

    print("Applied changes successfully")

replace_method()
