import re

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target = """    stage1_score = (
        (0.5 * ic_snr + 0.5 * mean_bucket_ic) * math.sqrt(max(coverage, 0.0))
        - 0.2 * float(events["bound_saturation"].mean() if len(events) else 0.0)
        - 0.2 * float((events["label"] == OUT_TO).mean() if len(events) else 1.0)
    )"""

replace = """    stage1_score = (
        (0.5 * ic_snr + 0.5 * mean_bucket_ic)
        - 0.2 * float(events["bound_saturation"].mean() if len(events) else 0.0)
        - 0.2 * float((events["label"] == OUT_TO).mean() if len(events) else 1.0)
    )"""

c = c.replace(target, replace)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
