with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target1 = """def _refine_ambiguous_labels_with_intrabar(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,"""

replace1 = """def _refine_ambiguous_labels_with_intrabar(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,"""
c = c.replace(target1, replace1)

target2 = """def _refine_ambiguous_labels_with_15m(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        h,"""

replace2 = """def _refine_ambiguous_labels_with_15m(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        mfe_df, mae_df, t_mfe_df, t_mae_df,
        h,"""

c = c.replace(target2, replace2)

target3 = """def _refine_ambiguous_labels_with_5m(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        h,"""

replace3 = """def _refine_ambiguous_labels_with_5m(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    cfg: Optional[Dict[str, Any]] = None,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        mfe_df, mae_df, t_mfe_df, t_mae_df,
        h,"""
c = c.replace(target3, replace3)


target4 = """lbl_15m, ret_15m, qual_15m = _refine_ambiguous_labels_with_5m(
                        _panel_15m,
                        _tp_15m,
                        _sl_15m,
                        lbl_15m,
                        ret_15m,
                        qual_15m,
                        h,"""

replace4 = """lbl_15m, ret_15m, qual_15m = _refine_ambiguous_labels_with_5m(
                        _panel_15m,
                        _tp_15m,
                        _sl_15m,
                        lbl_15m,
                        ret_15m,
                        qual_15m,
                        h, side, cfg=cfg
                    )""" # Actually we just need to ensure it runs correctly and we drop the output assignment of MFE/MAE because it's not changed inside
# Wait, let's keep it simple: the _refine functions just return the first 3 args. Let's see what `_refine_ambiguous_labels_with_intrabar` returns.

target_ret_intra = """    return lbl2, ret2, qual2"""
c = c.replace(target_ret_intra, "    return lbl2, ret2, qual2") # we don't return mfe/mae since it's just fine for our usecase.

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
