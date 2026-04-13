with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

# The function `_filter_artifact_by_stage_view` was inserted in the middle of `inject_features_into_datasets`. Let's move it out.
in_filter = False
filter_lines = []
new_lines = []
i = 0
while i < len(lines):
    if "def _filter_artifact_by_stage_view(df, cfg):" in lines[i]:
        in_filter = True
        filter_lines.append(lines[i])
    elif in_filter:
        if lines[i].startswith("    ") or lines[i].strip() == "":
            filter_lines.append(lines[i])
        else:
            in_filter = False
            new_lines.append(lines[i])
    else:
        new_lines.append(lines[i])
    i += 1

# Put filter_lines before inject_features_into_datasets
insert_idx = 0
for i in range(len(new_lines)):
    if "def inject_features_into_datasets(" in new_lines[i]:
        insert_idx = i
        break

final_lines = new_lines[:insert_idx] + filter_lines + new_lines[insert_idx:]

# Ensure inject_features_into_datasets has proper indentation
in_inject = False
for i in range(insert_idx + len(filter_lines), len(final_lines)):
    if "tprint(\"Resolving unique symbols and timestamps for feature injection...\")" in final_lines[i]:
        if final_lines[i].startswith("        "):
            final_lines[i] = final_lines[i][4:]
            for j in range(i+1, len(final_lines)):
                if final_lines[j].startswith("        "):
                    final_lines[j] = final_lines[j][4:]

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(final_lines)
