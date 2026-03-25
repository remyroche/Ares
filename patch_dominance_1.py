import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Remove _identify_dominated_rules method entirely
content = re.sub(
    r"    def _identify_dominated_rules\(self, df: pd\.DataFrame\) -> pd\.DataFrame:.*?(?=\n\n\n@njit\(cache=True, fastmath=True\))",
    "",
    content,
    flags=re.DOTALL
)

# Remove call to _identify_dominated_rules
content = re.sub(
    r"        summary_df = self\._identify_dominated_rules\(summary_df\)\n",
    "",
    content
)

# Remove dominated reporting block in score_registry_oos
content = re.sub(
    r"        dom_count = summary_df\.get\(\"dominated_by_parent\", pd\.Series\(False\)\)\.sum\(\)\n"
    r"        if dom_count > 0:\n"
    r"            tprint\(f\"Dominated rules flagged: \{dom_count\}\"\)\n"
    r"            top_dom = summary_df\[summary_df\[\"dominated_by_parent\"\]\]\.head\(5\)\n"
    r"            for _, row in top_dom\.iterrows\(\):\n"
    r"                tprint\(\n"
    r"                    f\"  - \{row\['canonical_key'\]\} dominated by \{row\['dominant_parent_key'\]\}\"\n"
    r"                \)\n",
    "",
    content
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
