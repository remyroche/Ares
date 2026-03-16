with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''    def _is_path_valid(self, conditions: List[RuleCondition]) -> Tuple[bool, str]:
        """
        Hardened validation: Group limits, contradictions, and polarity.
        """
        if not conditions:
            return False, "empty_path"

        interaction_groups = set()
        for c in conditions:
            m = self.metadata_lookup.get(c.feature_index)
            if not m:
                continue

            # Check if there's already a different feature from the same interaction group
            for existing_c in conditions:
                if existing_c.feature_index == c.feature_index:
                    continue

                existing_m = self.metadata_lookup.get(existing_c.feature_index)
                if existing_m and existing_m.interaction_group == m.interaction_group:
                    return False, f"interaction_group_violation_{m.interaction_group}"


        # Contradiction check (Normalized)
        feat_map = {}
        for c in conditions:
            if c.feature_index in feat_map:
                if feat_map[c.feature_index] != c.normalized_value:
                    return False, f"contradiction_{c.feature_name}"
            feat_map[c.feature_index] = c.normalized_value

        # Polarity Check: reject only all-negative paths
        # A path is all-negative if NO condition has normalized_value == 1
        if not any(c.normalized_value == 1 for c in conditions):
            return False, "all_negative_path"

        for c in conditions:
            if c.group in self.positive_only_groups and c.normalized_value != 1:
                return False, f"negative_not_allowed_{c.group}"

        positive_groups = {c.group for c in conditions if c.normalized_value == 1}
        missing_required = sorted(self.required_positive_groups - positive_groups)
        if missing_required:
            return False, f"missing_required_group_{missing_required[0]}"

        return True, "valid"''',
'''    def _is_path_valid(self, conditions: List[RuleCondition]) -> Tuple[bool, str]:
        """
        Hardened validation: Group limits, contradictions, and polarity.
        """
        if not conditions:
            return False, "empty_path"

        seen_groups = {}
        seen_features = {}

        for c in conditions:
            m = self.metadata_lookup.get(c.feature_index)
            if m is None:
                continue

            ig = m.interaction_group

            prev_feat = seen_groups.get(ig)
            if prev_feat is not None and prev_feat != c.feature_index:
                return False, f"interaction_group_violation_{ig}"

            seen_groups[ig] = c.feature_index

            prev_val = seen_features.get(c.feature_index)
            if prev_val is not None and prev_val != c.normalized_value:
                return False, f"contradiction_{c.feature_name}"

            seen_features[c.feature_index] = c.normalized_value

        # Polarity Check: reject only all-negative paths
        # A path is all-negative if NO condition has normalized_value == 1
        if not any(c.normalized_value == 1 for c in conditions):
            return False, "all_negative_path"

        for c in conditions:
            if c.group in self.positive_only_groups and c.normalized_value != 1:
                return False, f"negative_not_allowed_{c.group}"

        positive_groups = {c.group for c in conditions if c.normalized_value == 1}
        missing_required = sorted(self.required_positive_groups - positive_groups)
        if missing_required:
            return False, f"missing_required_group_{missing_required[0]}"

        return True, "valid"'''
)

# And P1-7:
code = code.replace(
'''    def _resolve_context_parent_mask(
        self, canonical_key: str, indices: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))
        parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
        if parent_key is None:
            return None
        ctx_name = self.parent_key_to_context_name.get(parent_key)
        if ctx_name is None:
            return None
        return self._slice_mask(self.context_lookup[ctx_name], indices)''',
'''    def _resolve_context_parent_mask(
        self, canonical_key: str, indices: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))
        parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
        if parent_key is None:
            return None
        ctx_name = self.parent_key_to_context_name.get(parent_key)
        if ctx_name is None:
            return None
        if not ctx_name.startswith("ctx__"):
            raise ValueError(f"Unexpected unresolved feature in canonical key: {ctx_name}")
        return self._slice_mask(self.context_lookup[ctx_name], indices)'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
