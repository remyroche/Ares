with open("extreme_price_movements/training.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "save_artifact_df(df, cfg[\"data_root\"], run_id, \"labels\", name)" in line:
        new_lines.append("        from extreme_price_movements.data_store import save_artifact_df\n")
        new_lines.append(line)
    elif "_save_event_index_artifact(p_evt, _pre_h[2], _pre_h[1], symbol_vocab)" in line:
        pass
    else:
        new_lines.append(line)

with open("extreme_price_movements/training.py", "w") as f:
    f.writelines(new_lines)
