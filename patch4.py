import re

with open("extreme_price_movements/simple_position_sizer.py", "r") as f:
    content = f.read()

# 1. Fix the isnan crash on string arrays in run_bucketed_simple_position_sizer
old_unique = "    unique_buckets = np.unique(bucket_labels[~np.isnan(bucket_labels)])"
new_unique = "    unique_buckets = np.unique(bucket_labels[~pd.isna(bucket_labels)])"
content = content.replace(old_unique, new_unique)

# 2. Fix the column filtering in run_simple_position_sizer_from_artifacts
# Currently it uses: head_cols = [c for c in active_df.columns if c.startswith("base_") or "pred" in c.lower() or "score" in c.lower()]
# We need it to be strictly tied to the strategy_id if applicable.

old_head_cols = """        # Identify columns to use as heads
        # Base models usually output things like base_H2, base_H4, etc. We will add them to feature_dict
        head_cols = [c for c in active_df.columns if c.startswith("base_") or "pred" in c.lower() or "score" in c.lower()]"""

new_head_cols = """        # Identify columns to use as heads
        # Base models usually output things like base_H2, base_H4, etc. We will add them to feature_dict
        # We must filter by strategy_id if present to satisfy: "uses ONLY the outputs from models trained under the same strategy_id"
        head_cols = []
        for c in active_df.columns:
            # Typical columns we want to evaluate
            if c.startswith("base_") or "pred" in c.lower() or "score" in c.lower() or "mae" in c.lower() or "mfe" in c.lower():
                # If the column has a strategy_id appended (e.g. base_H2_StratX), we MUST match it
                # If there's no strategy suffix in the column name, we allow it (e.g., standard base_H2)
                if strategy_id and strategy_id in c:
                    head_cols.append(c)
                elif not any(s.get("strategy_id", "") in c for s in strategies if s.get("strategy_id")):
                    # It's a generic column not tied to *any* other strategy
                    head_cols.append(c)"""

content = content.replace(old_head_cols, new_head_cols)

with open("extreme_price_movements/simple_position_sizer.py", "w") as f:
    f.write(content)
