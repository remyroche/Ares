with open('extreme_price_movements/features.py', 'r') as f:
    content = f.read()

content = content.replace(
    '''    # Add dynamically generated peer context and TS pct to skip set
    for k in feats.keys():
        if k.startswith("cs_rank_") or k.startswith("ts_pct_"):
            skip_transform_set.add(k)''',
    '''    # Add dynamically generated peer context and TS pct to skip set
    for k in feats.keys():
        if k.startswith("cs_rank_") or k.startswith("cs_rz_") or k.startswith("ts_pct_"):
            skip_transform_set.add(k)'''
)

with open('extreme_price_movements/features.py', 'w') as f:
    f.write(content)
