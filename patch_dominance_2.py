import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# 1. Remove from SCORER_REGISTRY_COLUMNS
content = re.sub(
    r"    \"dominant_parent_key\",\n    \"dominated_by_parent\",\n",
    "",
    content
)

# 2. RulePruner.prune_for_assessment
content = re.sub(
    r"            & \(df\[\"dominated_by_parent\"\] == False\)\n",
    "",
    content
)

# 3. IndependentRulePruner.prune
content = re.sub(
    r"        if \"dominated_by_parent\" not in df\.columns:\n            df\[\"dominated_by_parent\"\] = False\n\n",
    "",
    content
)
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

# 4. select_stage_a_contexts
content = re.sub(
    r"    registry\[\"reject_dominated\"\] = registry\.get\(\n        \"dominated_by_parent\", pd\.Series\(False, index=registry\.index\)\n    \)\n",
    "",
    content
)
content = re.sub(
    r"        \| registry\[\"reject_dominated\"\]\n",
    "",
    content
)
content = re.sub(
    r"        \"reject_dominated\",\n",
    "",
    content
)

# 5. build_stage_a_rejection_map (pruner gate items)
content = re.sub(
    r"                \(\n                    \"dominated_by_parent\",\n                    \"dominated_by_parent\",\n                    int\(\n                        consolidated_registry\.get\(\n                            \"dominated_by_parent\",\n                            pd\.Series\(False, index=consolidated_registry\.index\),\n                        \)\n                        \.fillna\(False\)\n                        \.sum\(\)\n                    \),\n                    \"must be False\",\n                \),\n",
    "",
    content
)
# build_stage_a_rejection_map (context_selector gate items)
content = re.sub(
    r"                \(\n                    \"reject_dominated\",\n                    \"dominated_by_parent\",\n                    selection_counts\.get\(\"reject_dominated\", 0\),\n                    \"must be False\",\n                \),\n",
    "",
    content
)

# 6. run_mining_stage (dominated_rule_summary.csv)
content = re.sub(
    r"    dom_df = scored_registry\[scored_registry\.get\(\"dominated_by_parent\", False\)\]\[\n        \[\"canonical_key\", \"dominant_parent_key\", \"composite_score\", \"hurdle_excess\"\]\n    \]\n    if not dom_df\.empty:\n        atomic_to_csv\(dom_df, output_dir \/ \"dominated_rule_summary\.csv\"\)\n\n",
    "",
    content
)

# 7. run_mining_stage (filter accepted_registry)
content = re.sub(
    r"    accepted_registry = accepted_registry\[\n        \~accepted_registry\.get\(\n            \"dominated_by_parent\", pd\.Series\(False, index=accepted_registry\.index\)\n        \)\.fillna\(False\)\n    \]\.copy\(\)\n",
    "    accepted_registry = accepted_registry.copy()\n",
    content
)

# 8. classify_rule_production_quality
content = re.sub(
    r"    dominated_by_parent = rule\.get\(\"dominated_by_parent\", False\)\n",
    "",
    content
)
content = re.sub(
    r"    # Check 7: Not dominated by parent \(warning only\)\n    if dominated_by_parent:\n        diagnostics\[\"warnings\"\].append\(\"dominated_by_simpler_parent\"\)\n\n",
    "",
    content
)

# 9. MaskAssessor.assess_rules (production classification rule building)
content = re.sub(
    r"                \"dominated_by_parent\": row\.get\(\"dominated_by_parent\", False\),\n",
    "",
    content
)


with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
