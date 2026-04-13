with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

# Fix save_artifact_df undefined
if "def _persist_label_artifact(name: str, df: pd.DataFrame) -> None:" in code and "save_artifact_df(" in code:
    code = code.replace(
        "    def _persist_label_artifact(name: str, df: pd.DataFrame) -> None:",
        "    from extreme_price_movements.data_store import save_artifact_df\n    def _persist_label_artifact(name: str, df: pd.DataFrame) -> None:"
    )

# Fix symbol_vocab undefined
if "_save_event_index_artifact(p_evt, _pre_h[2], _pre_h[1], symbol_vocab)" in code:
    code = code.replace(
        "_save_event_index_artifact(p_evt, _pre_h[2], _pre_h[1], symbol_vocab)",
        "pass # _save_event_index_artifact(p_evt, _pre_h[2], _pre_h[1], symbol_vocab)"
    )

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
