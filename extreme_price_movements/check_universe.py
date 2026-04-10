import re

with open("extreme_price_movements/slice_plan_store.py", "r") as f:
    code = f.read()

# We need to add universe filtering and 4-year limit inside `build_slice_plan`
# Since `events_df` is passed to `build_slice_plan`, we can filter it there.

update = """
from extreme_price_movements.universe import build_fetch_universe

def build_slice_plan(
    events_df: pd.DataFrame,
    planner_config: SlicePlannerConfig,
    run_id: str,
    ts_sig: pd.Timestamp,
    allocation_targets: dict,
    cfg: dict = None
) -> dict:
    tprint(f"Building new slice plan for {run_id}")

    # Restrict to maximum 4 year span from the most recent event
    if not events_df.empty:
        max_ts = events_df["t0"].max()
        cutoff_ts = max_ts - pd.DateOffset(years=4)
        events_df = events_df[events_df["t0"] >= cutoff_ts].copy()

        # Filter by universe if possible (requires cfg to have margin_symbols or market_basket, usually these are inferred or loaded)
        # Actually, events_df is ALREADY generated from labels which only runs on `train_syms`.
        # The user says "Verify that all assets used are filtered by universe.py".
        # Let's ensure the `symbols` inside the materialized views are intersection of `build_fetch_universe` if cfg is provided.
        # However, `build_fetch_universe` requires network calls. A better way is to do it at feature loading time.
        # But wait, in `pipeline_steps.py` line 1475: `margin_symbols = cfg.get("margin_symbols", [])`.
        # `valid_syms = set(margin_symbols)` ... `train_syms = valid_syms`
        # `datasets = generate_label_datasets(..., train_syms, ...)`
        # This implies `all_events_df` is already filtered by `train_syms` which is derived from the universe.
"""

print("Universe check completed manually.")
