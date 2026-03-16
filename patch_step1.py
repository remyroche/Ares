with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

# P2-8: Add comment to InteractionModel
code = code.replace(
'''class InteractionModel:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        allowed_group_pairs: Optional[Sequence[Tuple[str, str]]] = None
    ):''',
'''class InteractionModel:
    """
    LightGBM is trained without strict interaction constraints.
    Structural validity of rule paths is enforced in:
        RuleExtractor._is_path_valid()
    using interaction_group metadata.
    """
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        allowed_group_pairs: Optional[Sequence[Tuple[str, str]]] = None
    ):'''
)

# P0-1: Update _build_interaction_constraints
code = code.replace(
'''        # Remove explicit interaction constraints to allow interactions across families
        # and different groups according to the extractor rules.
        return None''',
'''        # Training is permissive; structural validity is enforced post-hoc
        # in RuleExtractor._is_path_valid().
        return []'''
)

# P0-1: Update get_constraint_summary
code = code.replace(
'''    def get_constraint_summary(self) -> Dict[str, Any]:
        """
        Returns a dictionary summarizing the interaction constraints.
        """
        # Map actual group names in constraints
        summary = collections.defaultdict(int)
        for c in self.constraints:
            if len(c) == 1:
                continue
            elif len(c) == 2:
                g1 = self.metadata[c[0]].group
                g2 = self.metadata[c[1]].group
                pair = tuple(sorted([g1, g2]))
                summary[f"{pair[0]}_{pair[1]}_pairs"] += 1
            elif len(c) == 3:
                g1 = self.metadata[c[0]].group
                g2 = self.metadata[c[1]].group
                g3 = self.metadata[c[2]].group
                triplet = tuple(sorted([g1, g2, g3]))
                summary[f"{triplet[0]}_{triplet[1]}_{triplet[2]}_triplets"] += 1
            else:
                groups_in_c = set(self.metadata[idx].group for idx in c)
                groups_str = "_".join(sorted(groups_in_c))
                summary[f"multi_group_{groups_str}"] += 1

        result = {
            'total_singletons': len(self.metadata),
            'total_constraints': len(self.constraints)
        }
        # Add group counts
        groups = set(m.group for m in self.metadata)
        for g in groups:
            result[f"num_{g}"] = sum(1 for m in self.metadata if m.group == g)

        result.update(summary)
        return result''',
'''    def get_constraint_summary(self) -> Dict[str, Any]:
        import collections
        result = {
            "total_singletons": len(self.metadata),
            "total_constraints": len(self.constraints) if self.constraints is not None else 0,
        }

        groups = set(m.group for m in self.metadata)
        for g in groups:
            result[f"num_{g}"] = sum(1 for m in self.metadata if m.group == g)

        if not self.constraints:
            return result

        summary = collections.defaultdict(int)
        for c in self.constraints:
            if len(c) == 1:
                summary["singleton"] += 1
                m = self.metadata[c[0]]
                summary[f"singleton_{m.group}"] += 1
            else:
                groups = set(self.metadata[i].group for i in c)
                if groups == {"regime"}:
                    summary["regime_cluster"] += 1
                elif groups == {"location"}:
                    summary["location_cluster"] += 1
                else:
                    summary["mixed_cluster"] += 1
        result.update(summary)
        return result'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
