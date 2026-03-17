import re

with open("extreme_price_movements/training.py", "r") as f:
    content = f.read()

# Replace _build_optimal_candidate_mask calls
content = re.sub(
    r"cand_mask, _ = _build_optimal_candidate_mask\((panel, feats, cfg)\)",
    r"cand_mask, _, mask_by_strategy = _build_optimal_candidate_mask(\1)",
    content
)
content = re.sub(
    r"cached_cand_mask, cfg = _build_optimal_candidate_mask\((panel, feats, cfg)\)",
    r"cached_cand_mask, cfg, mask_by_strategy = _build_optimal_candidate_mask(\1)",
    content
)

pattern_datasets = r"trade_sides = \[\"long\", \"short\"\]\n\s*kinds = \[\"mr\", \"tf\"\]\n\s*for side in trade_sides:\n\s*for k in kinds:"
replacement_datasets = r"""strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]
"""
content = re.sub(pattern_datasets, replacement_datasets, content)

content = content.replace("_cached_cand_mask=cached_cand_mask,", "_cached_cand_mask=mask_by_strategy.get(k, cached_cand_mask),")

pattern = r"trade_sides\s*=\s*\[\"long\",\s*\"short\"\]\s*\n\s*kinds\s*=\s*\[\"mr\",\s*\"tf\"\]\s*\n\s*for\s+side\s+in\s+trade_sides:\s*\n\s*for\s+k\s+in\s+kinds:"
replacement = r"""strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]
"""
content = re.sub(pattern, replacement, content)

pattern2 = r"trade_sides\s*=\s*\[\"long\",\s*\"short\"\]\s*\n\s*kinds\s*=\s*\[\"mr\",\s*\"tf\"\]\s*\n\s*for\s+side\s+in\s+trade_sides:\s*\n\s*for\s+kind\s+in\s+kinds:"
replacement2 = r"""strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        kind = strat["strategy_id"]
"""
content = re.sub(pattern2, replacement2, content)

pattern3 = r"for\s+side\s+in\s+trade_sides:\s*\n\s*for\s+k\s+in\s+kinds:"
replacement3 = r"""strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k = strat["strategy_id"]"""
content = re.sub(pattern3, replacement3, content)

pattern4 = r"for\s+side\s+in\s+trade_sides:\s*\n\s*for\s+kind\s+in\s+kinds:"
replacement4 = r"""strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        kind = strat["strategy_id"]"""
content = re.sub(pattern4, replacement4, content)

pattern5 = r"for\s+side\s+in\s+\[\"long\",\s*\"short\"\]:\s*\n\s*for\s+k_label\s+in\s+\[\"mr\",\s*\"tf\"\]:"
replacement5 = r"""strategies = get_strategies(cfg)
    for strat in strategies:
        side = strat["trade_side"]
        k_label = strat["strategy_id"]"""
content = re.sub(pattern5, replacement5, content)

# Also fix the final_models dictionary bug from my previous changes
# Instead of regex, I'll just find and replace the specific block if it exists
content = content.replace('cell_key = f"{kind.upper()}_{side}_H{H}"', 'cell_key = f"{kind}_H{H}"')
content = content.replace('f"label_min_tp_hit_rate_{kind_l}_{side_l}_h{int(h)}"', 'f"label_min_tp_hit_rate_{kind_l}_h{int(h)}"')
content = content.replace('f"label_min_tp_hit_rate_{kind_l}_{side_l}"', 'f"label_min_tp_hit_rate_{kind_l}"')
content = content.replace('f"label_min_tp_hit_rate_{side_l}_{kind_l}_h{int(h)}"', 'f"label_min_tp_hit_rate_{kind_l}_h{int(h)}"')
content = content.replace('f"label_min_tp_hit_rate_{side_l}_{kind_l}"', 'f"label_min_tp_hit_rate_{kind_l}"')
content = content.replace('f"label_min_tp_hit_rate_{str(kind).upper()}_{str(side).lower()}_H{int(h)}"', 'f"label_min_tp_hit_rate_{str(kind).upper()}_H{int(h)}"')
content = content.replace('f"barrier_tp_lo_{mode}_{kind_l}_{side_l}_h{int(h)}"', 'f"barrier_tp_lo_{mode}_{kind_l}_h{int(h)}"')
content = content.replace('f"barrier_tp_lo_{mode}_{kind_l}_{side_l}"', 'f"barrier_tp_lo_{mode}_{kind_l}"')
content = content.replace('f"barrier_tp_lo_{mode}_{side_l}_{kind_l}_h{int(h)}"', 'f"barrier_tp_lo_{mode}_{kind_l}_h{int(h)}"')
content = content.replace('f"barrier_tp_lo_{mode}_{side_l}_{kind_l}"', 'f"barrier_tp_lo_{mode}_{kind_l}"')
content = content.replace('f"barrier_tp_lo_{mode}_{str(kind).upper()}_{str(side).lower()}_H{int(h)}"', 'f"barrier_tp_lo_{mode}_{str(kind).upper()}_H{int(h)}"')


with open("extreme_price_movements/training.py", "w") as f:
    f.write(content)
