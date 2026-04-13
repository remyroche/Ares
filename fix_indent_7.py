with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    lines = f.readlines()

# It seems `inject_features_into_datasets` already exists and was modified manually but the bottom half of it got broken.
# Let's completely replace it with the correct version.
start_idx = -1
end_idx = -1

for i in range(len(lines)):
    if "def inject_features_into_datasets(datasets, ts_sig, cfg, req_keys):" in lines[i]:
        start_idx = i
    if start_idx != -1 and "return datasets" in lines[i]:
        # we found multiple returns, let's just make sure we capture the right one.
        pass

# let's just find the first inject_features_into_datasets and delete until the end of the file.
for i in range(len(lines)):
    if "def inject_features_into_datasets(datasets, ts_sig, cfg, req_keys):" in lines[i]:
        start_idx = i
        break

if start_idx != -1:
    lines = lines[:start_idx]

import sys
sys.path.append('.')
with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.writelines(lines)
