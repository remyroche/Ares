import re

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target1 = """def _refine_ambiguous_labels_with_intrabar(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,
    h: int,
    side: str,"""

# The problem is that h: int doesn't have a default but follows mfe_df which does.
replace1 = """def _refine_ambiguous_labels_with_intrabar(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    timeframe: str,
    has_local_cache_fn,
    load_or_download_fn,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,"""

# need to also fix where timeframe: str, etc. were in the original signature
target_full1 = """def _refine_ambiguous_labels_with_intrabar(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,
    h: int,
    side: str,
    timeframe: str,
    has_local_cache_fn,
    load_or_download_fn,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:"""

replace_full1 = """def _refine_ambiguous_labels_with_intrabar(
    panel: Dict[str, pd.DataFrame],
    tp_df: pd.DataFrame,
    sl_df: pd.DataFrame,
    lbl: pd.DataFrame,
    ret: pd.DataFrame,
    qual: pd.DataFrame,
    h: int,
    side: str,
    timeframe: str,
    has_local_cache_fn,
    load_or_download_fn,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:"""

c = c.replace(target_full1, replace_full1)


target_call1 = """    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        mfe_df, mae_df, t_mfe_df, t_mae_df,
        h,
        side,
        timeframe="15min",
        has_local_cache_fn=_has_local_15m_cache,
        load_or_download_fn=_load_or_download_15m,
        cfg=cfg,
    )"""

replace_call1 = """    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        h,
        side,
        timeframe="15min",
        has_local_cache_fn=_has_local_15m_cache,
        load_or_download_fn=_load_or_download_15m,
        mfe_df=mfe_df, mae_df=mae_df, t_mfe_df=t_mfe_df, t_mae_df=t_mae_df,
        cfg=cfg,
    )"""

c = c.replace(target_call1, replace_call1)


target_call2 = """    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        mfe_df, mae_df, t_mfe_df, t_mae_df,
        h,
        side,
        timeframe="5min",
        has_local_cache_fn=_has_local_5m_cache,
        load_or_download_fn=_load_or_download_5m,
        cfg=cfg,
    )"""

replace_call2 = """    return _refine_ambiguous_labels_with_intrabar(
        panel,
        tp_df,
        sl_df,
        lbl,
        ret,
        qual,
        h,
        side,
        timeframe="5min",
        has_local_cache_fn=_has_local_5m_cache,
        load_or_download_fn=_load_or_download_5m,
        mfe_df=mfe_df, mae_df=mae_df, t_mfe_df=t_mfe_df, t_mae_df=t_mae_df,
        cfg=cfg,
    )"""

c = c.replace(target_call2, replace_call2)


with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
