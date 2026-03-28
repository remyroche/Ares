import sys
import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Let's replace the logging of `top_final` in `run_mining_stage` to provide a thorough report as requested.
# But wait, the user asked to "make a thorough report for a top15 - using what scoring and validation rules?".
# A thorough report would include: score, hurdle_excess, side, display_arity, support_pct, uplift, directional_mean_ret, presence_freq, etc.

replacement = """        tprint("Top 15 Final Diverse Rules (Thorough Report):")
        top_final = select_top_diverse_rules(
            combined_global_registry, combined_mask_map, top_n=15
        )
        for i, (_, row) in enumerate(top_final.iterrows(), start=1):
            tprint(
                f"  {i:2d}. [{row.get('side', 'unknown').upper()}] {row['canonical_key']}\\n"
                f"      score={row.get('composite_score', 0):.3f} | "
                f"hurdle_excess={row.get('hurdle_excess', 0):.5f} | "
                f"support={row.get('mean_support_pct', 0):.2%} | "
                f"ret={row.get('directional_mean_ret', 0):.5f} | "
                f"uplift={row.get('mean_uplift', 0):.5f} | "
                f"presence={row.get('presence_freq', 0):.2%} | "
                f"sign_cons={row.get('sign_consistency', 0):.2%} | "
                f"arity={row.get('display_arity', 0)}"
            )"""

# Find the block:
#         tprint("Top 15 Final Diverse Rules:")
#         top_final = select_top_diverse_rules(
#             combined_global_registry, combined_mask_map, top_n=15
#         )
#         for i, (_, row) in enumerate(top_final.iterrows(), start=1):
#             tprint(
#                 f"  {i:2d}. {row['canonical_key']}: score={row['composite_score']:.3f}, hurdle_excess={row['hurdle_excess']:.5f}, arity={row['display_arity']}, side={row['side']}"
#             )

import re

# Since there can be slight variations, we'll use regex.
pattern = re.compile(r'        tprint\("Top 15 Final Diverse Rules:"\)\n        top_final = select_top_diverse_rules\(\n            combined_global_registry, combined_mask_map, top_n=15\n        \)\n        for i, \(_, row\) in enumerate\(top_final\.iterrows\(\), start=1\):\n            tprint\(\n                f"  \{i:2d\}\. \{row\[\'canonical_key\'\]\}: score=\{row\[\'composite_score\'\]:\.3f\}, hurdle_excess=\{row\[\'hurdle_excess\'\]:\.5f\}, arity=\{row\[\'display_arity\'\]\}, side=\{row\[\'side\'\]\}"\n            \)', re.MULTILINE)

match = pattern.search(content)
if not match:
    print("Could not find the block to replace.")
    # Fallback search
    pattern2 = re.compile(r'tprint\("Top 15 Final Diverse Rules:"\).*?side=\S+row\S+\'side\'\S+\}"\n\s+\)', re.DOTALL)
    match = pattern2.search(content)
    if not match:
        print("Fallback search failed too.")
        sys.exit(1)

new_content = content[:match.start()] + replacement + content[match.end():]
with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(new_content)

print("Patch 2 applied successfully")
