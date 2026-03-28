import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Instead of just `select_top_diverse_rules(..., top_n=15)`, we want to output a thorough report for a top15 in `run_lgbm_mask_generation_triad` at the very end.
# Actually, the user asked to "ensure we, in the end, generate a top10 with at least 6 longs/shorts, not too much repetition/overlap between the rules, and make a thorough report for a top15 - using what scoring and validation rules?"
# Let's add a top 15 extraction and thorough reporting step at the end of `run_lgbm_mask_generation_triad`.

new_code = """
    # --- Top 15 Diverse Rules Thorough Report ---
    if not merged_output["dedup_rules"].empty:
        tprint(f"\\n{'='*60}")
        tprint("TOP 15 DIVERSE RULES THOROUGH REPORT")
        tprint(f"{'='*60}")

        # We need a mask map for the dedup rules to run select_top_diverse_rules.
        # However, at this point, we don't have a combined_mask_map easily accessible unless we build it.
        # Alternatively, we can just use the deduplicated rules, and build masks for them.
        # But building masks requires X, which is only available inside run_mining_stage.

        # Let's modify run_mining_stage's output instead.
"""

print(content[5010:5050])
