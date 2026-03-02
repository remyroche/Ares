with open("extreme_price_movements/engine.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if "score_rows.append(_row)" in line and "return pd.DataFrame(score_rows)" in lines[i+2]:
        extra = """            # Also include meta features that might be needed downstream
            meta_cols = set(cfg.get("meta_feature_keys", [])) | set(cfg.get("mr_meta_feature_keys", [])) | set(cfg.get("tf_meta_feature_keys", []))
            for _cn in meta_cols:
                if _cn in grp.columns and _cn not in _row:
                    try:
                        _row[_cn] = float(grp.loc[idx, _cn])
                    except Exception:
                        _row[_cn] = 0.0

            # Ensure basic interaction columns and raw values are passed
            _extra_cols = ["vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct", "trend_t", "trend_z_t", "spike_score", "grind_score", "chop_score", "rv_12h", "rv_24h", "vol_z_base", "vol_z24_base", "ret6h", "trend_pct_base", "G_VOL", "G_TREND"]
            for _cn in _extra_cols:
                if _cn in grp.columns and _cn not in _row:
                    try:
                        _row[_cn] = float(grp.loc[idx, _cn])
                    except Exception:
                        _row[_cn] = 0.0
"""
        new_lines.insert(len(new_lines)-1, extra)

with open("extreme_price_movements/engine.py", "w") as f:
    f.writelines(new_lines)
