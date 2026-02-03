
def restore_execute():
    with open('layer2_full_codex.py', 'r') as f:
        codex_content = f.read()

    with open('src/training/steps/labeling/label_based_layer_2.py', 'r') as f:
        current_content = f.read()

    # Extract execute method from codex
    # It starts with async def execute and ends before register_label_based_layer_2_step usually

    start_marker = "async def execute(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:"
    start_idx = codex_content.find(start_marker)

    if start_idx == -1:
        # Try simplified signature
        start_marker = "async def execute("
        start_idx = codex_content.find(start_marker)

    if start_idx == -1:
        print("Could not find execute method in codex version")
        return

    # Find end of class or file
    # We can assume it goes until the end of the class.
    # But filtering the method text is safer.
    # Let's just append the execute method to the class in current file.

    # Extract the method block.
    # Count indentation to find end?
    # Or just take everything until the end of file excluding the registration function?

    reg_marker = "def register_label_based_layer_2_step() -> None:"
    end_idx = codex_content.find(reg_marker)

    if end_idx == -1:
        print("Could not find registration marker")
        return

    execute_block = codex_content[start_idx:end_idx]

    # Now insert into current content before the registration function
    reg_marker_curr = "def register_label_based_layer_2_step() -> None:"
    insert_idx = current_content.find(reg_marker_curr)

    if insert_idx == -1:
        print("Could not find insertion point in current file")
        return

    new_content = current_content[:insert_idx] + "\n    " + execute_block + "\n\n" + current_content[insert_idx:]

    # Fix potential double indentation if I pasted it raw (codex content already has indentation?)
    # start_idx in codex content is indented?
    # Let's check indentation of start_marker

    # Actually, let's just write the file
    with open('src/training/steps/labeling/label_based_layer_2.py', 'w') as f:
        f.write(new_content)

    print("Restored execute method")

restore_execute()
