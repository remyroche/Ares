import re
with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    content = f.read()

# 1. Add TargetNaNReason
reason_enum = """
class TargetNaNReason:
    HORIZON_EXCEEDED = "horizon_exceeded"
    BARRIER_UNRESOLVED = "barrier_unresolved"
    AMBIGUOUS_BAR = "ambiguous_bar"
    OUTSIDE_SUPPORT_MASK = "outside_support_mask"
    NEUTRAL_FILTERED = "neutral_filtered"
    OTHER_TARGET_NAN = "other_target_nan"

def generate_fwd_ret_with_reasons(panel: pd.DataFrame, fwd_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    \"\"\"Generates forward returns and a reason code array for Target NaNs.\"\"\"
    fwd_ret_wide = panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)

    reasons_wide = pd.DataFrame("", index=fwd_ret_wide.index, columns=fwd_ret_wide.columns)
    reasons_wide[np.isnan(fwd_ret_wide)] = TargetNaNReason.OTHER_TARGET_NAN
    if fwd_hours > 0 and len(reasons_wide) >= fwd_hours:
        reasons_wide.iloc[-fwd_hours:] = TargetNaNReason.HORIZON_EXCEEDED

    return fwd_ret_wide, reasons_wide
"""

if "class TargetNaNReason" not in content:
    idx = content.find("def _with_expected_columns")
    content = content[:idx] + reason_enum + "\n\n" + content[idx:]

# 2. Refactor fwd_ret_wide generation
old_gen = """    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )"""
new_gen = """    fwd_ret_wide, fwd_ret_reasons_wide = generate_fwd_ret_with_reasons(panel, fwd_hours)"""
content = content.replace(old_gen, new_gen)

# 3. Refactor fwd_ret_matrix
old_matrix = """    fwd_ret_matrix = fwd_ret_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=np.float32)
    target_signal = fwd_ret_matrix / np.maximum(np.sqrt(atr_pct_matrix), 1e-9)

    fwd_ret_norm_matrix = fwd_ret_matrix / np.maximum(atr_pct_matrix, 1e-9)
    fwd_ret_final = fwd_ret_matrix[time_idx, sym_idx]
    fwd_ret_norm_final = fwd_ret_norm_matrix[time_idx, sym_idx]"""

new_matrix = """    fwd_ret_matrix = fwd_ret_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=np.float32)
    fwd_ret_reasons_matrix = fwd_ret_reasons_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=object)

    target_signal = fwd_ret_matrix / np.maximum(np.sqrt(atr_pct_matrix), 1e-9)

    fwd_ret_norm_matrix = fwd_ret_matrix / np.maximum(atr_pct_matrix, 1e-9)
    fwd_ret_final = fwd_ret_matrix[time_idx, sym_idx]
    fwd_ret_norm_final = fwd_ret_norm_matrix[time_idx, sym_idx]
    fwd_ret_reasons_final = fwd_ret_reasons_matrix[time_idx, sym_idx]"""
content = content.replace(old_matrix, new_matrix)

# 4. apply_robust_data_filtering
old_def = """def apply_robust_data_filtering(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    overlap_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:"""

new_def = """def apply_robust_data_filtering(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    target_nan_reasons: Optional[np.ndarray] = None,
    overlap_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:"""
content = content.replace(old_def, new_def)

old_return1 = """            data,
            feature_dict,
            fwd_ret,
            fwd_ret_norm,
            {"dropped_features": [], "dropped_rows": 0},"""
new_return1 = """            data,
            feature_dict,
            fwd_ret,
            fwd_ret_norm,
            target_nan_reasons,
            {"dropped_features": [], "dropped_rows": 0},"""
content = content.replace(old_return1, new_return1)

old_middle = """    fwd_ret = fwd_ret[has_any_feature]
    fwd_ret_norm = fwd_ret_norm[has_any_feature]"""
new_middle = """    fwd_ret = fwd_ret[has_any_feature]
    fwd_ret_norm = fwd_ret_norm[has_any_feature]
    if target_nan_reasons is not None:
        target_nan_reasons = target_nan_reasons[has_any_feature]"""
content = content.replace(old_middle, new_middle)

old_end = """    fwd_ret_final = fwd_ret[final_keep_mask]
    fwd_ret_norm_final = fwd_ret_norm[final_keep_mask]

    meta = {"""
new_end = """    fwd_ret_final = fwd_ret[final_keep_mask]
    fwd_ret_norm_final = fwd_ret_norm[final_keep_mask]
    target_nan_reasons_final = target_nan_reasons[final_keep_mask] if target_nan_reasons is not None else None

    meta = {"""
content = content.replace(old_end, new_end)

old_final_ret = """    return data_final, features_final, fwd_ret_final, fwd_ret_norm_final, meta"""
new_final_ret = """    return data_final, features_final, fwd_ret_final, fwd_ret_norm_final, target_nan_reasons_final, meta"""
content = content.replace(old_final_ret, new_final_ret)

old_call = """    (
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        robust_meta,
    ) = apply_robust_data_filtering(
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        overlap_threshold=0.8,
    )"""

new_call = """    (
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        fwd_ret_reasons_final,
        robust_meta,
    ) = apply_robust_data_filtering(
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        target_nan_reasons=fwd_ret_reasons_final,
        overlap_threshold=0.8,
    )"""
content = content.replace(old_call, new_call)

old_triad_def = """def run_lgbm_mask_generation_triad(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    triad_targets: Dict[str, Dict[int, np.ndarray]],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:"""
new_triad_def = """def run_lgbm_mask_generation_triad(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    triad_targets: Dict[str, Dict[int, np.ndarray]],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    target_nan_reasons: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:"""
content = content.replace(old_triad_def, new_triad_def)

old_triad_call = """    run_lgbm_mask_generation_triad(
        data_final, feat_final, triad_targets, fwd_ret_final, fwd_ret_norm_final, cfg
    )"""
new_triad_call = """    run_lgbm_mask_generation_triad(
        data_final, feat_final, triad_targets, fwd_ret_final, fwd_ret_norm_final, fwd_ret_reasons_final, cfg
    )"""
content = content.replace(old_triad_call, new_triad_call)

old_side_def = """def run_side_pipeline(
    side: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    root_output_dir: Path,
    target_name: str = "primary_target",
    horizon: int = 0,
    bounded_target: Optional[np.ndarray] = None,
    bounded_target_surprisal: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:"""
new_side_def = """def run_side_pipeline(
    side: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    target_nan_reasons: Optional[np.ndarray],
    cfg: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    root_output_dir: Path,
    target_name: str = "primary_target",
    horizon: int = 0,
    bounded_target: Optional[np.ndarray] = None,
    bounded_target_surprisal: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:"""
content = content.replace(old_side_def, new_side_def)

old_side_call = """                side_results = run_side_pipeline(
                    side=side,
                    data=data,
                    feature_dict=feature_dict,
                    fwd_ret=fwd_ret,
                    fwd_ret_norm=fwd_ret_norm,
                    cfg=horizon_cfg,
                    folds=folds,
                    root_output_dir=horizon_target_dir,
                    target_name=target_name,
                    horizon=horizon,
                    bounded_target=bounded_target,
                    bounded_target_surprisal=bounded_target_surprisal,
                )"""
new_side_call = """                side_results = run_side_pipeline(
                    side=side,
                    data=data,
                    feature_dict=feature_dict,
                    fwd_ret=fwd_ret,
                    fwd_ret_norm=fwd_ret_norm,
                    target_nan_reasons=target_nan_reasons,
                    cfg=horizon_cfg,
                    folds=folds,
                    root_output_dir=horizon_target_dir,
                    target_name=target_name,
                    horizon=horizon,
                    bounded_target=bounded_target,
                    bounded_target_surprisal=bounded_target_surprisal,
                )"""
content = content.replace(old_side_call, new_side_call)

old_mining_for_target = """def run_mining_stage_for_target_horizon_side(
    side: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    root_output_dir: Path,
    target_name: str = "primary_target",
    horizon: int = 0,
    bounded_target: Optional[np.ndarray] = None,
    bounded_target_surprisal: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:"""
new_mining_for_target = """def run_mining_stage_for_target_horizon_side(
    side: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    target_nan_reasons: Optional[np.ndarray],
    cfg: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    root_output_dir: Path,
    target_name: str = "primary_target",
    horizon: int = 0,
    bounded_target: Optional[np.ndarray] = None,
    bounded_target_surprisal: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:"""
content = content.replace(old_mining_for_target, new_mining_for_target)

old_mining_for_target_call = """    stage_a_result = run_mining_stage_for_target_horizon_side(
        side=side,
        data=data,
        feature_dict=feature_dict,
        fwd_ret=fwd_ret,
        fwd_ret_norm=fwd_ret_norm,
        cfg=cfg,
        folds=folds,
        root_output_dir=root_output_dir,
        target_name=target_name,
        horizon=horizon,
        bounded_target=bounded_target,
        bounded_target_surprisal=bounded_target_surprisal,
    )"""
new_mining_for_target_call = """    stage_a_result = run_mining_stage_for_target_horizon_side(
        side=side,
        data=data,
        feature_dict=feature_dict,
        fwd_ret=fwd_ret,
        fwd_ret_norm=fwd_ret_norm,
        target_nan_reasons=target_nan_reasons,
        cfg=cfg,
        folds=folds,
        root_output_dir=root_output_dir,
        target_name=target_name,
        horizon=horizon,
        bounded_target=bounded_target,
        bounded_target_surprisal=bounded_target_surprisal,
    )"""
content = content.replace(old_mining_for_target_call, new_mining_for_target_call)

old_mining_stage = """def run_mining_stage(
    data: pd.DataFrame,
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    X: np.ndarray,
    metadata: List[FeatureMetadata],
    cfg: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
    allowed_group_pairs: Sequence[Tuple[str, str]],
    slot_order: Sequence[str] = ("trigger", "location", "regime"),
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    mask_resolver: Optional[
        Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
    ] = None,
    require_uplift: bool = False,
    rule_key_rewriter: Optional[
        Callable[[str], Tuple[Optional[str], Optional[str]]]
    ] = None,
    pipeline_stage_name: Optional[str] = None,
    explicit_side: Optional[str] = None,
    target_name: str = "primary_target",
    horizon: int = 0,
    primary_target_override: Optional[np.ndarray] = None,
    sample_weight_surprisal_override: Optional[np.ndarray] = None,
    run_step: str = "full",
    step1_input_dir: Optional[Path] = None,
    candidate_registry_override: Optional[pd.DataFrame] = None,
    bounded_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:"""
new_mining_stage = """def run_mining_stage(
    data: pd.DataFrame,
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    X: np.ndarray,
    metadata: List[FeatureMetadata],
    cfg: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
    allowed_group_pairs: Sequence[Tuple[str, str]],
    slot_order: Sequence[str] = ("trigger", "location", "regime"),
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    mask_resolver: Optional[
        Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
    ] = None,
    require_uplift: bool = False,
    rule_key_rewriter: Optional[
        Callable[[str], Tuple[Optional[str], Optional[str]]]
    ] = None,
    pipeline_stage_name: Optional[str] = None,
    explicit_side: Optional[str] = None,
    target_name: str = "primary_target",
    horizon: int = 0,
    primary_target_override: Optional[np.ndarray] = None,
    sample_weight_surprisal_override: Optional[np.ndarray] = None,
    run_step: str = "full",
    step1_input_dir: Optional[Path] = None,
    candidate_registry_override: Optional[pd.DataFrame] = None,
    bounded_target: Optional[np.ndarray] = None,
    target_nan_reasons: Optional[np.ndarray] = None,
) -> Dict[str, Any]:"""
content = content.replace(old_mining_stage, new_mining_stage)

old_mining_stage_call = """    return run_mining_stage(
        data=data,
        fwd_ret=side_fwd_ret,
        fwd_ret_norm=side_fwd_ret_norm,
        X=X_a,
        metadata=metadata_a,
        cfg=cfg,
        output_dir=stage_a_output_dir,
        stage_name=stage_a_spec.stage_name,
        allowed_group_pairs=stage_a_spec.allowed_group_pairs,
        slot_order=stage_a_spec.slot_order,
        folds=folds,
        mask_resolver=CanonicalRuleMaskResolver(X_a, metadata_a),
        pipeline_stage_name="stage_a_context",
        explicit_side=side,
        target_name=target_name,
        horizon=horizon,
        primary_target_override=side_target,
        sample_weight_surprisal_override=bounded_target_surprisal,
        run_step=cfg.get("run_step", "full"),
        step1_input_dir=cfg.get("step1_dir") if cfg.get("run_step") == "step2" else None,
        bounded_target=bounded_target,
    )"""
new_mining_stage_call = """    return run_mining_stage(
        data=data,
        fwd_ret=side_fwd_ret,
        fwd_ret_norm=side_fwd_ret_norm,
        X=X_a,
        metadata=metadata_a,
        cfg=cfg,
        output_dir=stage_a_output_dir,
        stage_name=stage_a_spec.stage_name,
        allowed_group_pairs=stage_a_spec.allowed_group_pairs,
        slot_order=stage_a_spec.slot_order,
        folds=folds,
        mask_resolver=CanonicalRuleMaskResolver(X_a, metadata_a),
        pipeline_stage_name="stage_a_context",
        explicit_side=side,
        target_name=target_name,
        horizon=horizon,
        primary_target_override=side_target,
        sample_weight_surprisal_override=bounded_target_surprisal,
        run_step=cfg.get("run_step", "full"),
        step1_input_dir=cfg.get("step1_dir") if cfg.get("run_step") == "step2" else None,
        bounded_target=bounded_target,
        target_nan_reasons=target_nan_reasons,
    )"""
content = content.replace(old_mining_stage_call, new_mining_stage_call)

old_assess = """    def assess_rules(
        self,
        registry: pd.DataFrame,
        X: np.ndarray,
        data: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        target_ret_by_side: Dict[str, np.ndarray],
        n_day_buckets: int,
        day_codes: np.ndarray,
        total_symbol_days: float,
        target_name: str = "primary_target",
        horizon: int = 0,
        explicit_side: Optional[str] = None,
    ) -> pd.DataFrame:"""
new_assess = """    def assess_rules(
        self,
        registry: pd.DataFrame,
        X: np.ndarray,
        data: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        target_ret_by_side: Dict[str, np.ndarray],
        n_day_buckets: int,
        day_codes: np.ndarray,
        total_symbol_days: float,
        target_name: str = "primary_target",
        horizon: int = 0,
        explicit_side: Optional[str] = None,
        target_nan_reasons: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:"""
content = content.replace(old_assess, new_assess)

old_assess_call = """    stage_a_result = assessor.assess_rules(
        registry=candidate_registry_override,
        X=X,
        data=data,
        fwd_ret=fwd_ret,
        folds=folds,
        target_ret_by_side=target_ret_by_side,
        n_day_buckets=n_day_buckets,
        day_codes=day_codes,
        total_symbol_days=total_symbol_days,
        target_name=target_name,
        horizon=horizon,
        explicit_side=explicit_side,
    )"""
new_assess_call = """    stage_a_result = assessor.assess_rules(
        registry=candidate_registry_override,
        X=X,
        data=data,
        fwd_ret=fwd_ret,
        folds=folds,
        target_ret_by_side=target_ret_by_side,
        n_day_buckets=n_day_buckets,
        day_codes=day_codes,
        total_symbol_days=total_symbol_days,
        target_name=target_name,
        horizon=horizon,
        explicit_side=explicit_side,
        target_nan_reasons=target_nan_reasons,
    )"""
content = content.replace(old_assess_call, new_assess_call)


# Update _compute_subset_ridge_details
old_def = """    def _compute_subset_ridge_details(
        self, X, fwd_ret, mask, folds, tp_f: np.ndarray = None
    ) -> Dict[str, Any]:"""
new_def = """    def _compute_subset_ridge_details(
        self, X, fwd_ret, mask, folds, tp_f: np.ndarray = None, target_nan_reasons: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:"""
content = content.replace(old_def, new_def)

old_empty = """            return {
                "subset_auc": np.nan,
                "subset_roc_auc": np.nan,
                "subset_pr_auc": np.nan,
                "top_quartile_precision": np.nan,
                "coverage": 0.0,
                "oof_preds": np.full(len(X), np.nan, dtype=np.float32),
                "folds_used": 0,
                "folds_skipped": 0,
            }"""
new_empty = """            return {
                "subset_auc": np.nan,
                "subset_roc_auc": np.nan,
                "subset_pr_auc": np.nan,
                "top_quartile_precision": np.nan,
                "coverage": 0.0,
                "oof_preds": np.full(len(X), np.nan, dtype=np.float32),
                "folds_used": 0,
                "folds_skipped": 0,
                "mask_oof_corr": np.nan,
                "mask_oof_r2": np.nan,
                "fold_sign_consistency": np.nan,
                "positive_fold_fraction": np.nan,
                "negative_fold_fraction": np.nan,
                "decile_monotonic_spearman": np.nan,
                "top_decile_mean_target": np.nan,
                "bottom_decile_mean_target": np.nan,
                "decile_spread_mean": np.nan,
            }"""
content = content.replace(old_empty, new_empty)

old_log_tr = """            if n_tr_before > 0 and n_tr_after < n_tr_before:
                n_tr_target_nan = np.sum(~np.isfinite(y_tr))
                n_tr_feat_nan = np.sum(np.isfinite(y_tr) & (~np.all(np.isfinite(X_tr), axis=1)))

                tprint(
                    f"WARNING: Fold {fold_id} Ridge training: Dropped {n_tr_before - n_tr_after}/{n_tr_before} "
                    f"({100*(1-n_tr_after/n_tr_before):.1f}%) samples. "
                    f"[Target NaN: {n_tr_target_nan}, Feature NaN: {n_tr_feat_nan}]"
                )"""

new_log_tr = """            if n_tr_before > 0 and n_tr_after < n_tr_before:
                target_nan_mask = ~np.isfinite(y_tr)
                n_tr_target_nan = np.sum(target_nan_mask)
                n_tr_feat_nan = np.sum((~target_nan_mask) & (~np.all(np.isfinite(X_tr), axis=1)))

                target_reasons = {"horizon_exceeded": 0, "barrier_unresolved": 0, "ambiguous_bar": 0, "outside_support_mask": 0, "neutral_filtered": 0, "other_target_nan": 0}
                if target_nan_reasons is not None and n_tr_target_nan > 0:
                    fold_reasons = target_nan_reasons[tr_masked][target_nan_mask]
                    unique_reasons, counts = np.unique(fold_reasons, return_counts=True)
                    for reason, count in zip(unique_reasons, counts):
                        if reason in target_reasons:
                            target_reasons[reason] += count
                        else:
                            target_reasons["other_target_nan"] += count
                else:
                    target_reasons["other_target_nan"] += n_tr_target_nan

                tprint(
                    f"WARNING: Fold {fold_id} Ridge training: Dropped {n_tr_before - n_tr_after}/{n_tr_before} "
                    f"({100*(1-n_tr_after/n_tr_before):.1f}%) samples. "
                    f"[Target NaN: {n_tr_target_nan}, Feature NaN: {n_tr_feat_nan}] "
                    f"TargetNaNReasons[horizon_exceeded={target_reasons['horizon_exceeded']}, barrier_unresolved={target_reasons['barrier_unresolved']}, ambiguous_bar={target_reasons['ambiguous_bar']}, outside_support_mask={target_reasons['outside_support_mask']}, neutral_filtered={target_reasons['neutral_filtered']}, other_target_nan={target_reasons['other_target_nan']}]"
                )"""
content = content.replace(old_log_tr, new_log_tr)

old_log_va = """            if n_va_before > 0 and n_va_after < n_va_before:
                n_va_target_nan = np.sum(~np.isfinite(y_va))
                n_va_feat_nan = np.sum(np.isfinite(y_va) & (~np.all(np.isfinite(X_va), axis=1)))
                tprint(
                    f"WARNING: Fold {fold_id} Ridge validation: Dropped {n_va_before - n_va_after}/{n_va_before} "
                    f"({100*(1-n_va_after/n_va_before):.1f}%) samples. "
                    f"[Target NaN: {n_va_target_nan}, Feature NaN: {n_va_feat_nan}]"
                )"""

new_log_va = """            if n_va_before > 0 and n_va_after < n_va_before:
                target_nan_mask = ~np.isfinite(y_va)
                n_va_target_nan = np.sum(target_nan_mask)
                n_va_feat_nan = np.sum((~target_nan_mask) & (~np.all(np.isfinite(X_va), axis=1)))

                target_reasons = {"horizon_exceeded": 0, "barrier_unresolved": 0, "ambiguous_bar": 0, "outside_support_mask": 0, "neutral_filtered": 0, "other_target_nan": 0}
                if target_nan_reasons is not None and n_va_target_nan > 0:
                    fold_reasons = target_nan_reasons[va_masked][target_nan_mask]
                    unique_reasons, counts = np.unique(fold_reasons, return_counts=True)
                    for reason, count in zip(unique_reasons, counts):
                        if reason in target_reasons:
                            target_reasons[reason] += count
                        else:
                            target_reasons["other_target_nan"] += count
                else:
                    target_reasons["other_target_nan"] += n_va_target_nan

                tprint(
                    f"WARNING: Fold {fold_id} Ridge validation: Dropped {n_va_before - n_va_after}/{n_va_before} "
                    f"({100*(1-n_va_after/n_va_before):.1f}%) samples. "
                    f"[Target NaN: {n_va_target_nan}, Feature NaN: {n_va_feat_nan}] "
                    f"TargetNaNReasons[horizon_exceeded={target_reasons['horizon_exceeded']}, barrier_unresolved={target_reasons['barrier_unresolved']}, ambiguous_bar={target_reasons['ambiguous_bar']}, outside_support_mask={target_reasons['outside_support_mask']}, neutral_filtered={target_reasons['neutral_filtered']}, other_target_nan={target_reasons['other_target_nan']}]"
                )"""
content = content.replace(old_log_va, new_log_va)

old_return2 = """        total_elapsed = time.perf_counter() - subset_auc_start
        if total_elapsed >= 0.20:
            tprint(
                "Stage A: Ridge subset AUC internals "
                f"folds_used={folds_used} folds_skipped={folds_skipped} "
                f"filter_elapsed={fold_filter_time:.2f}s fit_predict_elapsed={fit_predict_time:.2f}s "
                f"total_elapsed={total_elapsed:.2f}s"
            )
        return {"""

new_return2 = """
        # --- Compute Expanded Learnability Metrics ---
        mask_oof_corr = np.nan
        mask_oof_r2 = np.nan
        fold_sign_consistency = np.nan
        positive_fold_fraction = np.nan
        negative_fold_fraction = np.nan
        decile_monotonic_spearman = np.nan
        top_decile_mean_target = np.nan
        bottom_decile_mean_target = np.nan
        decile_spread_mean = np.nan

        target_nan_total_train = 0
        target_nan_total_val = 0
        train_target_nan_reasons = {"horizon_exceeded": 0, "barrier_unresolved": 0, "ambiguous_bar": 0, "outside_support_mask": 0, "neutral_filtered": 0, "other_target_nan": 0}
        val_target_nan_reasons = {"horizon_exceeded": 0, "barrier_unresolved": 0, "ambiguous_bar": 0, "outside_support_mask": 0, "neutral_filtered": 0, "other_target_nan": 0}

        for tr_idx, va_idx in folds:
            tr_masked = tr_idx[mask[tr_idx]]
            va_masked = va_idx[mask[va_idx]]

            y_tr = y[tr_masked]
            target_nan_mask_tr = ~np.isfinite(y_tr)
            target_nan_total_train += np.sum(target_nan_mask_tr)

            if target_nan_reasons is not None and np.sum(target_nan_mask_tr) > 0:
                fold_reasons = target_nan_reasons[tr_masked][target_nan_mask_tr]
                u_reasons, counts = np.unique(fold_reasons, return_counts=True)
                for r, c in zip(u_reasons, counts):
                    if r in train_target_nan_reasons:
                        train_target_nan_reasons[r] += c
                    else:
                        train_target_nan_reasons["other_target_nan"] += c
            else:
                train_target_nan_reasons["other_target_nan"] += np.sum(target_nan_mask_tr)

            y_va = y[va_masked]
            target_nan_mask_va = ~np.isfinite(y_va)
            target_nan_total_val += np.sum(target_nan_mask_va)

            if target_nan_reasons is not None and np.sum(target_nan_mask_va) > 0:
                fold_reasons = target_nan_reasons[va_masked][target_nan_mask_va]
                u_reasons, counts = np.unique(fold_reasons, return_counts=True)
                for r, c in zip(u_reasons, counts):
                    if r in val_target_nan_reasons:
                        val_target_nan_reasons[r] += c
                    else:
                        val_target_nan_reasons["other_target_nan"] += c
            else:
                val_target_nan_reasons["other_target_nan"] += np.sum(target_nan_mask_va)


        valid_mask = mask.astype(bool) & np.isfinite(y) & np.isfinite(oof_preds)
        effective_rows = int(np.sum(valid_mask))

        if effective_rows > 0:
            preds_valid = oof_preds[valid_mask]
            targets_valid = y[valid_mask]

            # B1. Masked OOF Correlation
            if np.std(preds_valid) > 0 and np.std(targets_valid) > 0 and len(preds_valid) > 1:
                mask_oof_corr = np.corrcoef(preds_valid, targets_valid)[0, 1]

            # B2. Masked R2
            mean_y = np.mean(targets_valid)
            ss_tot = np.sum((targets_valid - mean_y) ** 2)
            ss_res = np.sum((targets_valid - preds_valid) ** 2)
            if ss_tot > 0:
                mask_oof_r2 = 1.0 - (ss_res / ss_tot)
            else:
                tprint("WARNING: Cannot compute mask_oof_r2 due to zero variance in targets.")

            # B3. Fold sign consistency
            fold_means = []
            for fold_id, (tr_idx, va_idx) in enumerate(folds):
                va_masked_idx = va_idx[mask[va_idx]]
                valid_fold_mask = np.isfinite(y[va_masked_idx]) & np.isfinite(oof_preds[va_masked_idx])
                fold_targets = y[va_masked_idx][valid_fold_mask]
                if len(fold_targets) > 0:
                    fold_means.append(np.mean(fold_targets))

            if fold_means:
                n_pos = sum(1 for m in fold_means if m > 0)
                n_neg = sum(1 for m in fold_means if m < 0)
                n_nonzero = n_pos + n_neg

                positive_fold_fraction = n_pos / len(fold_means)
                negative_fold_fraction = n_neg / len(fold_means)

                if n_nonzero > 0:
                    fold_sign_consistency = max(n_pos, n_neg) / n_nonzero
                else:
                    tprint("WARNING: Cannot compute fold_sign_consistency due to zero fold means.")

            # B4. Decile monotonicity
            if effective_rows >= 10:
                import scipy.stats
                order = np.argsort(preds_valid)
                binned = np.array_split(order, 10)
                decile_means = []
                for b in binned:
                    if len(b) > 0:
                        decile_means.append(np.mean(targets_valid[b]))

                if len(decile_means) >= 5:
                    top_decile_mean_target = decile_means[-1]
                    bottom_decile_mean_target = decile_means[0]
                    decile_spread_mean = top_decile_mean_target - bottom_decile_mean_target

                    spearman_corr, _ = scipy.stats.spearmanr(np.arange(1, len(decile_means) + 1), decile_means)
                    if np.isfinite(spearman_corr):
                        decile_monotonic_spearman = float(spearman_corr)
                else:
                    tprint("WARNING: Cannot compute decile_monotonic_spearman due to insufficient populated deciles.")
            else:
                tprint("WARNING: Cannot compute decile_monotonic_spearman due to insufficient effective rows.")

        total_elapsed = time.perf_counter() - subset_auc_start
        if total_elapsed >= 0.20:
            tprint(
                "Stage A: Ridge subset AUC internals "
                f"folds_used={folds_used} folds_skipped={folds_skipped} "
                f"filter_elapsed={fold_filter_time:.2f}s fit_predict_elapsed={fit_predict_time:.2f}s "
                f"total_elapsed={total_elapsed:.2f}s"
            )
        return {"""
content = content.replace(old_return2, new_return2)

old_ret_dict = """            "folds_used": int(folds_used),
            "folds_skipped": int(folds_skipped),
        }"""

new_ret_dict = """            "folds_used": int(folds_used),
            "folds_skipped": int(folds_skipped),
            "mask_oof_corr": float(mask_oof_corr) if np.isfinite(mask_oof_corr) else np.nan,
            "mask_oof_r2": float(mask_oof_r2) if np.isfinite(mask_oof_r2) else np.nan,
            "fold_sign_consistency": float(fold_sign_consistency) if np.isfinite(fold_sign_consistency) else np.nan,
            "positive_fold_fraction": float(positive_fold_fraction) if np.isfinite(positive_fold_fraction) else np.nan,
            "negative_fold_fraction": float(negative_fold_fraction) if np.isfinite(negative_fold_fraction) else np.nan,
            "decile_monotonic_spearman": float(decile_monotonic_spearman) if np.isfinite(decile_monotonic_spearman) else np.nan,
            "top_decile_mean_target": float(top_decile_mean_target) if np.isfinite(top_decile_mean_target) else np.nan,
            "bottom_decile_mean_target": float(bottom_decile_mean_target) if np.isfinite(bottom_decile_mean_target) else np.nan,
            "decile_spread_mean": float(decile_spread_mean) if np.isfinite(decile_spread_mean) else np.nan,
            "target_nan_total_train": target_nan_total_train,
            "target_nan_total_val": target_nan_total_val,
            "train_target_nan_reasons": train_target_nan_reasons,
            "val_target_nan_reasons": val_target_nan_reasons,
        }"""
content = content.replace(old_ret_dict, new_ret_dict)

old_assess_extract = """                    "rule_type_class": rule_type_class,
                }
            )

        assessment_df = pd.DataFrame(assessment_results)"""
new_assess_extract = """                    "rule_type_class": rule_type_class,
                    "mask_oof_corr": ridge_details.get("mask_oof_corr", np.nan),
                    "mask_oof_r2": ridge_details.get("mask_oof_r2", np.nan),
                    "fold_sign_consistency": ridge_details.get("fold_sign_consistency", np.nan),
                    "positive_fold_fraction": ridge_details.get("positive_fold_fraction", np.nan),
                    "negative_fold_fraction": ridge_details.get("negative_fold_fraction", np.nan),
                    "decile_monotonic_spearman": ridge_details.get("decile_monotonic_spearman", np.nan),
                    "top_decile_mean_target": ridge_details.get("top_decile_mean_target", np.nan),
                    "bottom_decile_mean_target": ridge_details.get("bottom_decile_mean_target", np.nan),
                    "decile_spread_mean": ridge_details.get("decile_spread_mean", np.nan),
                    "target_nan_total_train": ridge_details.get("target_nan_total_train", 0),
                    "target_nan_total_val": ridge_details.get("target_nan_total_val", 0),
                }
            )

            if "train_target_nan_reasons" in ridge_details and ridge_details["train_target_nan_reasons"] is not None:
                for k, v in ridge_details["train_target_nan_reasons"].items():
                    assessment_results[-1][f"train_target_nan_{k}"] = v
            if "val_target_nan_reasons" in ridge_details and ridge_details["val_target_nan_reasons"] is not None:
                for k, v in ridge_details["val_target_nan_reasons"].items():
                    assessment_results[-1][f"val_target_nan_{k}"] = v

        # Print the per-candidate target-drop summary message
        if "train_target_nan_reasons" in ridge_details:
            train_reasons = ridge_details.get("train_target_nan_reasons", {})
            val_reasons = ridge_details.get("val_target_nan_reasons", {})

            # Helper to format the dict
            def _format_reasons(r_dict):
                return f"horizon_exceeded={r_dict.get('horizon_exceeded', 0)}, barrier_unresolved={r_dict.get('barrier_unresolved', 0)}, ambiguous_bar={r_dict.get('ambiguous_bar', 0)}, outside_support_mask={r_dict.get('outside_support_mask', 0)}, neutral_filtered={r_dict.get('neutral_filtered', 0)}, other_target_nan={r_dict.get('other_target_nan', 0)}"

            # Merged reasons for print log
            merged_reasons = {k: train_reasons.get(k, 0) + val_reasons.get(k, 0) for k in set(train_reasons) | set(val_reasons)}

            tprint(
                f"Stage A: Ridge target-drop summary key={canonical_key} "
                f"train_target_nan={ridge_details.get('target_nan_total_train', 0)} "
                f"val_target_nan={ridge_details.get('target_nan_total_val', 0)} "
                f"reasons[{_format_reasons(merged_reasons)}]"
            )

        assessment_df = pd.DataFrame(assessment_results)"""

content = content.replace(old_assess_extract, new_assess_extract)

old_log_done = """        if (row_idx + 1) % log_every == 0 or row_idx == len(registry) - 1:
            tprint(
                f"Stage A: Ridge assessment done {row_idx + 1}/{len(registry)} "
                f"key={canonical_key} "
                f"pnl={ridge_pnl_raw:.6f} "
                f"trades_per_day={ridge_trade_metrics.get('avg_trades_per_day', 0):.2f} "
                f"coverage={ridge_details['coverage']:.4f} "
                f"elapsed={time.perf_counter() - rule_start_ts:.2f}s"
            )"""

new_log_done = """        if (row_idx + 1) % log_every == 0 or row_idx == len(registry) - 1:
            tprint(
                f"Stage A: Ridge learnability done {row_idx + 1}/{len(registry)} "
                f"key={canonical_key} "
                f"mask_oof_corr={ridge_details.get('mask_oof_corr', np.nan):.6f} "
                f"mask_oof_r2={ridge_details.get('mask_oof_r2', np.nan):.6f} "
                f"fold_sign_consistency={ridge_details.get('fold_sign_consistency', np.nan):.3f} "
                f"decile_monotonic_spearman={ridge_details.get('decile_monotonic_spearman', np.nan):.3f} "
                f"coverage={ridge_details.get('coverage', 0.0):.4f} "
                f"elapsed={time.perf_counter() - rule_start_ts:.2f}s"
            )"""

content = content.replace(old_log_done, new_log_done)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(content)
