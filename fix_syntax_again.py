import re

with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    code = f.read()

# Fix syntax error at: health_feats = load_features_for_stage_or_all(cfg, ts_sig, cfg[\"data_root\"],
code = code.replace(r'cfg[\"data_root\"]', 'cfg["data_root"]')

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.write(code)


with open("extreme_price_movements/run_pipeline.py", "r") as f:
    code = f.read()

# Fix unexpected indentation at:
#     feats = _smart_load_features_selected(cfg,
#         root_dir=cfg["data_root"],
code = code.replace("    feats = _smart_load_features_selected(cfg, ", "    feats = load_features_for_stage_or_all(cfg, ")

with open("extreme_price_movements/run_pipeline.py", "w") as f:
    f.write(code)
