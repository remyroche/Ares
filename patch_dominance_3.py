import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Let's just run regex to remove any leftover lines referencing "dominated_by_parent"
# or "dominated_by_parent_rejected" or "reject_dominated" or "dominated" if they are isolated.
# Instead, let's look at the remaining occurrences and fix them properly.

content = re.sub(
    r"                \(\n                                \"dominated_by_parent\",\n                    selection_counts\.get\(\"reject_dominated\", 0\),\n                    \"must be False\",\n                \),\n",
    "",
    content
)

# And one in IndependentRulePruner if it was missed:
content = re.sub(
    r"            \"dominated_by_parent_rejected\": int\(df\[\"dominated_by_parent\"\].sum\(\)\),\n",
    "",
    content
)
content = re.sub(
    r"            & \(\~df\[\"dominated_by_parent\"\].fillna\(False\).astype\(bool\)\)\n",
    "",
    content
)
content = re.sub(
    r"            f\"Dominated=\{gate_summary\['dominated_by_parent_rejected'\]\} \| \"\n",
    "",
    content
)


with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
