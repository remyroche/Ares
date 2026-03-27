with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target1 = """                lbl, ret, qual = compute_triple_barrier_labels("""
replace1 = """                lbl, ret, qual, mfe_df, mae_df, t_mfe_df, t_mae_df = compute_triple_barrier_labels("""
c = c.replace(target1, replace1)

# Now, we need to extract from cache
target2 = """                lbl, ret, qual = layer2_cache[key2]"""
replace2 = """                v2 = layer2_cache[key2]
                lbl, ret, qual = v2[0], v2[1], v2[2]
                mfe_df = v2[3] if len(v2) >= 7 else pd.DataFrame(0.0, index=lbl.index, columns=lbl.columns)
                mae_df = v2[4] if len(v2) >= 7 else pd.DataFrame(0.0, index=lbl.index, columns=lbl.columns)
                t_mfe_df = v2[5] if len(v2) >= 7 else pd.DataFrame(0.0, index=lbl.index, columns=lbl.columns)
                t_mae_df = v2[6] if len(v2) >= 7 else pd.DataFrame(0.0, index=lbl.index, columns=lbl.columns)"""
c = c.replace(target2, replace2)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
