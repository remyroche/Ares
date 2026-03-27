import re
with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target1 = """                lbl_15m, ret_15m, qual_15m = _refine_ambiguous_labels_with_5m(
                        _panel_15m,
                        _tp_15m,
                        _sl_15m,
                        lbl_15m,
                        ret_15m,
                        qual_15m,
                        h,
                        side,
                        cfg=cfg,
                    )"""

replace1 = """                lbl_15m, ret_15m, qual_15m = _refine_ambiguous_labels_with_5m(
                        _panel_15m,
                        _tp_15m,
                        _sl_15m,
                        lbl_15m,
                        ret_15m,
                        qual_15m,
                        h,
                        side,
                        cfg=cfg,
                    )"""

c = c.replace(target1, replace1)

# Now, we need to add the flat arrays to stack_cache

target2 = """                label_arr = lbl.to_numpy(dtype=np.float32, copy=False).ravel()
                payoff_arr = ret.to_numpy(dtype=np.float32, copy=False).ravel()
                qual_arr = qual.to_numpy(dtype=np.float32, copy=False).ravel()"""

replace2 = """                label_arr = lbl.to_numpy(dtype=np.float32, copy=False).ravel()
                payoff_arr = ret.to_numpy(dtype=np.float32, copy=False).ravel()
                qual_arr = qual.to_numpy(dtype=np.float32, copy=False).ravel()
                mfe_arr_flat = mfe_df.to_numpy(dtype=np.float32, copy=False).ravel()
                mae_arr_flat = mae_df.to_numpy(dtype=np.float32, copy=False).ravel()
                t_mfe_arr_flat = t_mfe_df.to_numpy(dtype=np.float32, copy=False).ravel()
                t_mae_arr_flat = t_mae_df.to_numpy(dtype=np.float32, copy=False).ravel()"""

c = c.replace(target2, replace2)

target3 = """                stack_cache[stack_key] = (
                    stacked_idx,
                    panel_idx_arr,
                    label_arr[valid_mask_flat],
                    payoff_arr[valid_mask_flat],
                    qual_arr[valid_mask_flat],
                    tp_arr[valid_mask_flat],
                    sl_arr[valid_mask_flat],
                    _h_eff_arr[valid_mask_flat] if _h_eff_arr is not None else _h_arr_scalar,
                )"""

replace3 = """                stack_cache[stack_key] = (
                    stacked_idx,
                    panel_idx_arr,
                    label_arr[valid_mask_flat],
                    payoff_arr[valid_mask_flat],
                    qual_arr[valid_mask_flat],
                    tp_arr[valid_mask_flat],
                    sl_arr[valid_mask_flat],
                    _h_eff_arr[valid_mask_flat] if _h_eff_arr is not None else _h_arr_scalar,
                    mfe_arr_flat[valid_mask_flat],
                    mae_arr_flat[valid_mask_flat],
                    t_mfe_arr_flat[valid_mask_flat],
                    t_mae_arr_flat[valid_mask_flat],
                )"""

c = c.replace(target3, replace3)


target4 = """                    stacked_idx,
                    panel_idx_arr,
                    label_arr,
                    payoff_arr,
                    qual_arr,
                    tp_arr,
                    sl_arr,
                    h_arr,
                ) = stack_cache[stack_key]"""

replace4 = """                    stacked_idx,
                    panel_idx_arr,
                    label_arr,
                    payoff_arr,
                    qual_arr,
                    tp_arr,
                    sl_arr,
                    h_arr,
                    mfe_arr,
                    mae_arr,
                    t_mfe_arr,
                    t_mae_arr,
                ) = stack_cache[stack_key]"""

c = c.replace(target4, replace4)


target5 = """                    "sl": sl_arr,
                    "horizon_eff": h_arr,
                    "__panel_idx__": panel_idx_arr,
                },"""

replace5 = """                    "sl": sl_arr,
                    "horizon_eff": h_arr,
                    "__panel_idx__": panel_idx_arr,
                    "mfe": mfe_arr,
                    "mae": mae_arr,
                    "t_mfe": t_mfe_arr,
                    "t_mae": t_mae_arr,
                },"""

c = c.replace(target5, replace5)


with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
