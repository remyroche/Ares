with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

# Modification 2: update compare_tbm_parameters.py cache layer format
# Let's see how `compute_triple_barrier_labels` is called

target1 = """                lbl_fast, _, _ = compute_triple_barrier_labels("""
replace1 = """                lbl_fast, _, _, _, _, _, _ = compute_triple_barrier_labels("""
c = c.replace(target1, replace1)

target2 = """                    lbl_15m, ret_15m, qual_15m = compute_triple_barrier_labels("""
replace2 = """                    lbl_15m, ret_15m, qual_15m, _, _, _, _ = compute_triple_barrier_labels("""
c = c.replace(target2, replace2)

target3 = """                    lbl, ret, qual = compute_triple_barrier_labels("""
replace3 = """                    lbl, ret, qual, mfe_df, mae_df, t_mfe_df, t_mae_df = compute_triple_barrier_labels("""
c = c.replace(target3, replace3)

target4 = """                layer2_cache[key2] = (lbl, ret, qual)"""
replace4 = """                layer2_cache[key2] = (lbl, ret, qual, mfe_df, mae_df, t_mfe_df, t_mae_df)"""
c = c.replace(target4, replace4)

target5 = """            # Layer 2: (lbl, ret, qual)
            v2 = layer2_cache[key]
            lbl = v2[0]
            ret = v2[1]
            qual = v2[2] if len(v2) >= 3 else None"""
replace5 = """            # Layer 2: (lbl, ret, qual, mfe, mae, t_mfe, t_mae)
            v2 = layer2_cache[key]
            lbl = v2[0]
            ret = v2[1]
            qual = v2[2] if len(v2) >= 3 else None
            mfe_df = v2[3] if len(v2) >= 7 else None
            mae_df = v2[4] if len(v2) >= 7 else None
            t_mfe_df = v2[5] if len(v2) >= 7 else None
            t_mae_df = v2[6] if len(v2) >= 7 else None"""
c = c.replace(target5, replace5)


target6 = """        for v in layer2_cache.values():
        if not isinstance(v, tuple) or len(v) != 3:
            # compute_triple_barrier_labels returns 3 elements (lbl, ret, qual)
            if len(v) == 2:  # Legacy support
                pass
            else:
                continue"""
replace6 = """    for v in layer2_cache.values():
        if not isinstance(v, tuple) or len(v) < 2:
            continue"""
c = c.replace(target6, replace6)


target7 = """            lbl = pd.read_parquet(cache_dir / item["label_file"])
            ret = pd.read_parquet(cache_dir / item["return_file"])

            # Legacy quality handling
            qual = None"""
replace7 = """            lbl = pd.read_parquet(cache_dir / item["label_file"])
            ret = pd.read_parquet(cache_dir / item["return_file"])

            try:
                mfe_df = pd.read_parquet(cache_dir / item.get("mfe_file", ""))
                mae_df = pd.read_parquet(cache_dir / item.get("mae_file", ""))
                t_mfe_df = pd.read_parquet(cache_dir / item.get("t_mfe_file", ""))
                t_mae_df = pd.read_parquet(cache_dir / item.get("t_mae_file", ""))
            except Exception:
                mfe_df = mae_df = t_mfe_df = t_mae_df = None

            # Legacy quality handling
            qual = None"""
c = c.replace(target7, replace7)


target8 = """            geom = {"bound_saturation": float(item.get("bound_saturation", 0.0))}
            layer1_loaded[key] = (tp_df, sl_df, geom, dyn_h)
            layer2_loaded[key] = (lbl, ret, qual)"""
replace8 = """            geom = {"bound_saturation": float(item.get("bound_saturation", 0.0))}
            layer1_loaded[key] = (tp_df, sl_df, geom, dyn_h)
            layer2_loaded[key] = (lbl, ret, qual, mfe_df, mae_df, t_mfe_df, t_mae_df)"""
c = c.replace(target8, replace8)

target9 = """            entry["quality_file"] = qual_file

            entries.append(entry)"""
replace9 = """            entry["quality_file"] = qual_file

            if len(v2) >= 7 and v2[3] is not None:
                mfe_file = f"{stem}_mfe.parquet"
                mae_file = f"{stem}_mae.parquet"
                t_mfe_file = f"{stem}_t_mfe.parquet"
                t_mae_file = f"{stem}_t_mae.parquet"
                v2[3].to_parquet(cache_dir / mfe_file, compression="zstd")
                v2[4].to_parquet(cache_dir / mae_file, compression="zstd")
                v2[5].to_parquet(cache_dir / t_mfe_file, compression="zstd")
                v2[6].to_parquet(cache_dir / t_mae_file, compression="zstd")
                entry["mfe_file"] = mfe_file
                entry["mae_file"] = mae_file
                entry["t_mfe_file"] = t_mfe_file
                entry["t_mae_file"] = t_mae_file

            entries.append(entry)"""
c = c.replace(target9, replace9)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
