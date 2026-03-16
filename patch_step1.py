with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

# 1. Modify collapse_duplicate_groups for stage_a_context
code = code.replace(
'''    positive_only_groups: Tuple[str, ...] = ()
    required_positive_groups: Tuple[str, ...] = ()
    collapse_duplicate_groups: Tuple[str, ...] = ()
    if pipeline_stage_name == "stage_a_context":
        collapse_duplicate_groups = ("location", "regime")''',
'''    positive_only_groups: Tuple[str, ...] = ()
    required_positive_groups: Tuple[str, ...] = ()
    collapse_duplicate_groups: Tuple[str, ...] = ()
    if pipeline_stage_name == "stage_a_context":
        collapse_duplicate_groups = ("location",)'''
)

# 2. Modify reconstruct_stage_b_key
code = code.replace(
'''    def reconstruct_stage_b_key(raw_key: str) -> Tuple[Optional[str], Optional[str]]:
        slots = raw_key.split("|")
        trigger_slot = "(*)"
        parent_context_key = None
        for slot in slots:
            slot_value = slot.strip("()")
            if slot_value == "*" or "==" not in slot_value:
                continue
            feature_name = slot_value.split("==")[0]
            if feature_name in INTRADAY_TRIGGER_COLUMNS:
                trigger_slot = slot
            elif feature_name.startswith("ctx__"):
                parent_context_key = context_to_key.get(feature_name)
        if parent_context_key is None or trigger_slot == "(*)":
            return None, None
        parent_slots = parent_context_key.split("|")
        return f"{trigger_slot}|{parent_slots[1]}|{parent_slots[2]}", parent_context_key''',
'''    def reconstruct_stage_b_key(raw_key: str) -> Tuple[Optional[str], Optional[str]]:
        slots = raw_key.split("|")
        trigger_conditions = []
        parent_context_key = None
        for slot in slots:
            slot_value = slot.strip("()")
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    continue
                feature_name = cond_str.split("==")[0]
                if feature_name in INTRADAY_TRIGGER_COLUMNS:
                    trigger_conditions.append(cond_str)
                elif feature_name.startswith("ctx__"):
                    parent_context_key = context_to_key.get(feature_name)

        if parent_context_key is None or not trigger_conditions:
            return None, None

        trigger_slot = f"({'&'.join(sorted(trigger_conditions))})"
        parent_slots = parent_context_key.split("|")
        return f"{trigger_slot}|{parent_slots[1]}|{parent_slots[2]}", parent_context_key'''
)

# 3. Global consolidator check
code = code.replace(
'''    global_scorer = RuleScorer(metadata_a + metadata_b, cfg, mask_resolver=combined_resolver)
    global_consolidator = RuleConsolidator(
        metadata_a + metadata_b,
        cfg,
        mask_resolver=combined_resolver,
        scorer=global_scorer,
    )''',
'''    global_scorer = RuleScorer(metadata_a + metadata_b, cfg, mask_resolver=combined_resolver)
    use_economic_consolidator = cfg.get("use_economic_consolidator", True)
    if use_economic_consolidator:
        global_consolidator = EconomicRuleConsolidator(
            metadata_a + metadata_b,
            cfg,
            mask_resolver=combined_resolver,
            scorer=global_scorer,
        )
    else:
        global_consolidator = RuleConsolidator(
            metadata_a + metadata_b,
            cfg,
            mask_resolver=combined_resolver,
            scorer=global_scorer,
        )'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
