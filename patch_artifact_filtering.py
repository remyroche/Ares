import re

with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    code = f.read()

# Filter loaded dataframes in run_training_step:
# `df_local = load_artifact_df(cfg["data_root"], run_id, "labels", name)` -> we should filter `df_local`

filter_helper = """
def _filter_artifact_by_stage_view(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    view = cfg.get("_active_stage_view")
    if not view or df is None or df.empty:
        return df

    if "symbols" in view and view["symbols"] is not None:
        sym_col = "__symbol__" if "__symbol__" in df.columns else "symbol" if "symbol" in df.columns else None
        if sym_col:
            df = df[df[sym_col].isin(view["symbols"])]

    if view.get("allowed_start_ts") or view.get("allowed_end_ts"):
        ts_col = "__ts__" if "__ts__" in df.columns else "timestamp" if "timestamp" in df.columns else "t0" if "t0" in df.columns else None
        if ts_col:
            if view.get("allowed_start_ts"):
                df = df[pd.to_datetime(df[ts_col], utc=True) >= pd.to_datetime(view["allowed_start_ts"])]
            if view.get("allowed_end_ts"):
                df = df[pd.to_datetime(df[ts_col], utc=True) <= pd.to_datetime(view["allowed_end_ts"])]
    return df
"""

if "_filter_artifact_by_stage_view" not in code:
    # Insert near the top
    code = code.replace("from extreme_price_movements.utils import (\n    get_tbm_hyperparams,", "from extreme_price_movements.utils import (\n    get_tbm_hyperparams,\n)\n" + filter_helper + "\n")


# Update run_training_step
code = re.sub(
    r'(df_local = load_artifact_df\(cfg\["data_root"\], run_id, "labels", name\))',
    r'\1\n        df_local = _filter_artifact_by_stage_view(df_local, cfg)',
    code
)

# Update run_base_hpo_step
code = re.sub(
    r'(df = load_artifact_df\(data_root, run_id, "labels", name\))',
    r'\1\n                df = _filter_artifact_by_stage_view(df, cfg)',
    code
)

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.write(code)


# Also do this for run_ridge_sizer.py which loads meta oof predictions
with open("extreme_price_movements/run_ridge_sizer.py", "r") as f:
    code = f.read()

if "_filter_artifact_by_stage_view" not in code:
    code = code.replace("from extreme_price_movements.utils import tprint", "from extreme_price_movements.utils import tprint\n" + filter_helper)

code = re.sub(
    r'(oof_preds = load_strategy_oofs\(data_root, run_id, strategy_id\))',
    r'\1\n            oof_preds = _filter_artifact_by_stage_view(oof_preds, dict(CFG, _active_stage_view=cfg.get("_active_stage_view")))',
    code
)

# load_trade_outcomes
code = re.sub(
    r'(trade_outcomes = load_trade_outcomes\(data_root, run_id, oof_preds\))',
    r'\1\n                trade_outcomes = _filter_artifact_by_stage_view(trade_outcomes, dict(CFG, _active_stage_view=cfg.get("_active_stage_view")))',
    code
)

with open("extreme_price_movements/run_ridge_sizer.py", "w") as f:
    f.write(code)
