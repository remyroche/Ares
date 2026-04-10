with open("extreme_price_movements/pipeline_steps.py", "r") as f:
    code = f.read()

# I messed up the injection point of the helper.
import re
# Let's write a simple regex replacement

helper = """
def _filter_artifact_by_stage_view(df, cfg):
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
            df_ts = pd.to_datetime(df[ts_col], utc=True)
            if view.get("allowed_start_ts"):
                df = df[df_ts >= pd.to_datetime(view["allowed_start_ts"])]
                df_ts = pd.to_datetime(df[ts_col], utc=True)
            if view.get("allowed_end_ts"):
                df = df[df_ts <= pd.to_datetime(view["allowed_end_ts"])]
    return df
"""

if "def _filter_artifact_by_stage_view" not in code:
    code = code.replace("from extreme_price_movements.utils import tprint", "from extreme_price_movements.utils import tprint\n" + helper)

# Do the same for run_ridge_sizer
with open("extreme_price_movements/run_ridge_sizer.py", "r") as f:
    code_ridge = f.read()

if "def _filter_artifact_by_stage_view" not in code_ridge:
    code_ridge = code_ridge.replace("from extreme_price_movements.utils import tprint", "from extreme_price_movements.utils import tprint\n" + helper)

with open("extreme_price_movements/run_ridge_sizer.py", "w") as f:
    f.write(code_ridge)

with open("extreme_price_movements/pipeline_steps.py", "w") as f:
    f.write(code)
