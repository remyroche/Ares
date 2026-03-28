import re

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

target = """def _refine_ambiguous_labels_with_intrabar(
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

replace = """def _refine_ambiguous_labels_with_intrabar(
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
    cfg: Optional[Dict[str, Any]] = None,
    mfe_df: pd.DataFrame = None,
    mae_df: pd.DataFrame = None,
    t_mfe_df: pd.DataFrame = None,
    t_mae_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:"""

c = c.replace(target, replace)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
