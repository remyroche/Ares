with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

# The function _filter_artifact_by_stage_view got indented accidentally, un-indent it
in_func = False
for i in range(len(lines)):
    if "def _filter_artifact_by_stage_view(df, cfg):" in lines[i]:
        in_func = True

    if in_func:
        # Check if line starts with 4 spaces and remove them
        if lines[i].startswith("    "):
            lines[i] = lines[i][4:]
        else:
            # Empty line or something else
            lines[i] = lines[i].lstrip()

        # Determine end of function (when we hit another def with 0 or 4 spaces)
        if i < len(lines) - 1 and lines[i+1].lstrip().startswith("def ") and lines[i+1].startswith(("def ", "    def ")):
            in_func = False

# also remove the unused TRAP_FEATURE_KEYS and GAMMA_FEATURE_KEYS things in inject_features_into_datasets
for i in range(len(lines)):
    if "dataset_features[name] = set(TRAP_FEATURE_KEYS)" in lines[i] or "dataset_features[name] = set(GAMMA_FEATURE_KEYS)" in lines[i]:
        lines[i] = "                dataset_features[name] = set(req_keys)\n"
    if "heartbeat_every = " in lines[i]:
        lines[i] = ""

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
