with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target1 = """                    lbl, ret, qual = _refine_ambiguous_labels_with_15m(
                        panel_for_labeling,
                        _sampled_tp,
                        _sampled_sl,
                        lbl,
                        ret,
                        qual,
                        h,
                        side,
                        cfg=cfg,
                    )"""

replace1 = """                    lbl, ret, qual = _refine_ambiguous_labels_with_15m(
                        panel_for_labeling,
                        _sampled_tp,
                        _sampled_sl,
                        lbl,
                        ret,
                        qual,
                        h,
                        side,
                        cfg=cfg,
                        mfe_df=mfe_df, mae_df=mae_df, t_mfe_df=t_mfe_df, t_mae_df=t_mae_df
                    )"""
c = c.replace(target1, replace1)

target2 = """                lbl, ret, qual = _refine_ambiguous_labels_with_15m(
                    panel_for_labeling,
                    tp_df_subsampled,
                    sl_df_subsampled,
                    lbl,
                    ret,
                    qual,
                    h,
                    side,
                    cfg=cfg,
                )"""

replace2 = """                lbl, ret, qual = _refine_ambiguous_labels_with_15m(
                    panel_for_labeling,
                    tp_df_subsampled,
                    sl_df_subsampled,
                    lbl,
                    ret,
                    qual,
                    h,
                    side,
                    cfg=cfg,
                    mfe_df=mfe_df, mae_df=mae_df, t_mfe_df=t_mfe_df, t_mae_df=t_mae_df
                )"""
c = c.replace(target2, replace2)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
