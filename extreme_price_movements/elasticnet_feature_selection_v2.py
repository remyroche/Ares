import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import ElasticNet
from scipy.stats import spearmanr
from extreme_price_movements.purged_cv import PurgedKFold

def _backfill_to_floor(local_idx, freqs, floor, subset_size):
    selected = list(local_idx)
    target = min(floor, subset_size)
    if len(selected) >= target:
        return np.array(sorted(selected), dtype=int)
    avail = [i for i in range(subset_size) if i not in selected]
    avail = sorted(avail, key=lambda i: freqs[i], reverse=True)
    selected.extend(avail[: target - len(selected)])
    return np.array(sorted(set(selected)), dtype=int)

def _compute_inner_splits(timestamps: Optional[np.ndarray], n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < max(20, n_splits * 5):
        mid = max(1, n_samples // 2)
        if mid >= n_samples:
            return []
        return [(np.arange(0, mid), np.arange(mid, n_samples))]

    cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    splits = []
    for tr, va in cv.split(dummy_X):
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr, va))

    if not splits:
        mid = max(1, n_samples // 2)
        if mid >= n_samples:
            return []
        return [(np.arange(0, mid), np.arange(mid, n_samples))]
    return splits

def compute_pairwise_jaccard(feature_sets: List[np.ndarray]) -> float:
    if len(feature_sets) < 2:
        return 1.0
    jaccards = []
    for i in range(len(feature_sets)):
        for j in range(i + 1, len(feature_sets)):
            s1 = set(feature_sets[i])
            s2 = set(feature_sets[j])
            if not s1 and not s2:
                jaccards.append(1.0)
            else:
                jaccards.append(len(s1.intersection(s2)) / len(s1.union(s2)))
    return float(np.mean(jaccards))

def score_fold(y_true: np.ndarray, y_pred: np.ndarray, model_kind: str) -> float:
    """Returns metric where higher is better."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    if not np.any(valid):
        return -1e9

    y_t = y_true[valid]
    y_p = y_pred[valid]

    if model_kind == "edge":
        # Top-decile realized net + spearman tie break
        k = max(1, int(len(y_p) * 0.10))
        idx = np.argpartition(y_p, -k)[-k:]
        base = float(np.mean(y_t[idx]))
        sp, _ = spearmanr(y_p, y_t)
        sp = float(sp) if pd.notna(sp) else 0.0
        return base + 0.05 * sp

    elif model_kind == "downside":
        # Negative MAE + false safe penalty
        mae = float(np.mean(np.abs(y_p - y_t)))

        # false safe: pred in lowest 20%, true in highest 20%
        safe_t = np.percentile(y_p, 20)
        danger_t = np.percentile(y_t, 80)
        pred_safe = y_p <= safe_t
        if np.sum(pred_safe) > 0:
            fsr = np.sum(pred_safe & (y_t >= danger_t)) / np.sum(pred_safe)
        else:
            fsr = 0.0

        return -(mae + 0.5 * fsr)

    elif model_kind == "uncertainty":
        # Correlation - underestimation penalty
        sp, _ = spearmanr(y_p, y_t)
        sp = float(sp) if pd.notna(sp) else 0.0

        # underestimation: pred in lowest 20%, true in highest 20%
        low_unc_t = np.percentile(y_p, 20)
        high_err_t = np.percentile(y_t, 80)
        pred_low = y_p <= low_unc_t
        if np.sum(pred_low) > 0:
            uer = np.sum(pred_low & (y_t >= high_err_t)) / np.sum(pred_low)
        else:
            uer = 0.0

        return sp - 0.25 * uer

    return 0.0

def select_features_via_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray],
    model_kind: str,
    feature_names: List[str],
    alpha_grid: np.ndarray,
    l1_ratio_grid: List[float],
    sample_weight_train: Optional[np.ndarray] = None,
    inner_n_splits: int = 3,
    max_features_cap: Optional[int] = None,
    min_features_floor: int = 5,
    sparsity_penalty: float = 0.04,
    selection_freq_threshold: float = 0.67,
    use_sign_consistency: bool = False,
) -> Dict:

    n_samples, n_features = X_train.shape
    if max_features_cap is None:
        max_features_cap = n_features

    target_floor = min(min_features_floor, max_features_cap, n_features)

    if model_kind == "edge":
        sparsity_penalty_eff = 0.04
    elif model_kind == "downside":
        sparsity_penalty_eff = 0.035
    elif model_kind == "uncertainty":
        sparsity_penalty_eff = 0.025
    else:
        sparsity_penalty_eff = sparsity_penalty

    if inner_n_splits <= 2:
        eff_sel_thresh = 0.50
    elif inner_n_splits == 3:
        eff_sel_thresh = 2.0 / 3.0
    else:
        eff_sel_thresh = selection_freq_threshold

    splits = _compute_inner_splits(timestamps_train, n_samples, inner_n_splits)
    if not splits:
        mid = max(1, n_samples // 2)
        if mid < n_samples:
            splits = [(np.arange(0, mid), np.arange(mid, n_samples))]

    path_results = []

    # Pre-scale standardizations per fold to save time
    from sklearn.preprocessing import StandardScaler
    fold_data = []
    for tr, va in splits:
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(np.nan_to_num(X_train[tr]))
        x_va_s = scaler.transform(np.nan_to_num(X_train[va]))
        fold_data.append({
            "tr": tr, "va": va,
            "x_tr_s": x_tr_s, "x_va_s": x_va_s,
            "y_tr": y_train[tr], "y_va": y_train[va],
            "w_tr": sample_weight_train[tr] if sample_weight_train is not None else None
        })

    for l1_ratio in l1_ratio_grid:
        for alpha in alpha_grid:
            fold_scores = []
            fold_selected = []

            all_zero_count = 0

            fold_coef_signs = []
            for fd in fold_data:
                model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=3000, tol=1e-3, random_state=42, selection="cyclic")
                try:
                    model.fit(fd["x_tr_s"], fd["y_tr"], sample_weight=fd["w_tr"])
                    preds = model.predict(fd["x_va_s"])
                    score = score_fold(fd["y_va"], preds, model_kind)
                    coefs = model.coef_
                    nonzero_idx = np.where(np.abs(coefs) > 1e-9)[0]
                    signs = np.sign(coefs)
                except Exception:
                    score = -1e9
                    nonzero_idx = np.array([], dtype=int)
                    signs = np.zeros(n_features)

                fold_scores.append(score)
                fold_selected.append(nonzero_idx)
                fold_coef_signs.append(signs)

                if len(nonzero_idx) == 0:
                    all_zero_count += 1

            union_selected = set()
            for sel in fold_selected:
                union_selected.update(sel)

            n_sel = len(union_selected)
            if n_sel > max_features_cap:
                continue

            freqs = np.zeros(n_features)
            for sel in fold_selected:
                freqs[sel] += 1
            freqs /= len(splits)

            union_list = list(union_selected)
            mean_sel_freq = float(np.mean(freqs[union_list])) if union_list else 0.0
            sel_freq_min = float(np.min(freqs[union_list])) if union_list else 0.0
            sel_freq_max = float(np.max(freqs[union_list])) if union_list else 0.0

            mean_jaccard = compute_pairwise_jaccard(fold_selected)

            msc = None
            if use_sign_consistency and union_list:
                signs_mat = np.vstack(fold_coef_signs)
                consistencies = []
                for idx in union_list:
                    col = signs_mat[:, idx]
                    nz_col = col[col != 0]
                    if len(nz_col) > 0:
                        frac_pos = np.sum(nz_col > 0) / len(nz_col)
                        frac_neg = np.sum(nz_col < 0) / len(nz_col)
                        consistencies.append(max(frac_pos, frac_neg))
                if consistencies:
                    msc = float(np.mean(consistencies))
                    stab_score = 0.55 * mean_sel_freq + 0.25 * mean_jaccard + 0.20 * msc
                else:
                    stab_score = 0.65 * mean_sel_freq + 0.35 * mean_jaccard
            else:
                stab_score = 0.65 * mean_sel_freq + 0.35 * mean_jaccard

            norm_n = n_sel / max(1, n_features)
            adj_score = stab_score - sparsity_penalty_eff * norm_n

            path_results.append({
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "n_selected": n_sel,
                "selected_union": union_list,
                "mean_sel_freq": mean_sel_freq,
                "mean_jaccard": mean_jaccard,
                "stability_score": stab_score,
                "freqs": freqs,
                "path_zero_fold_count": all_zero_count,
                "selection_freq_min": sel_freq_min,
                "selection_freq_mean": mean_sel_freq,
                "selection_freq_max": sel_freq_max,
                "mean_sign_consistency": msc,
                "normalized_n_selected": norm_n,
                "adjusted_score": adj_score
            })

            # Prune path: if all folds selected 0 features, no need to try larger alphas for this l1_ratio
            if all_zero_count == len(splits):
                break

    if not path_results:
        fallback_n = min(n_features, max(1, min_features_floor))
        fallback_idx = np.arange(fallback_n, dtype=int)
        fallback_names = [feature_names[i] for i in fallback_idx]

        return {
            "selected_idx": fallback_idx,
            "selected_names": fallback_names,
            "best_alpha": 0.0,
            "best_l1_ratio": 0.0,
            "chosen_alpha": 0.0,
            "chosen_l1_ratio": 0.0,
            "best_score": 0.0,
            "score_std": 0.0,
            "threshold_score": 0.0,
            "n_features_selected": len(fallback_idx),
            "selection_frequency": np.zeros(n_features),
            "selected_feature_frequencies": np.zeros(len(fallback_idx)),
            "mean_jaccard": 0.0,
            "mean_sign_consistency": None,
            "stability_score": 0.0,
            "adjusted_score": 0.0,
            "effective_selection_freq_threshold": eff_sel_thresh,
            "model_feature_floor": target_floor,
            "was_backfilled_to_floor": True,
            "path_table": pd.DataFrame(),
        }

    df = pd.DataFrame(path_results)

    # 1 std rule + stability
    best_idx = df["mean_score"].idxmax()
    best_score = df.loc[best_idx, "mean_score"]
    best_std = df.loc[best_idx, "std_score"]
    thresh = best_score - best_std

    candidates = df[df["mean_score"] >= thresh].copy()

    # Use effective sparsity penalty
    candidates["normalized_n_selected"] = candidates["n_selected"] / max(1, n_features)
    candidates["adjusted_score"] = candidates["stability_score"] - sparsity_penalty_eff * candidates["normalized_n_selected"]

    # Sort by: highest stability-adjusted score -> sparsest -> larger alpha
    candidates = candidates.sort_values(
        by=["adjusted_score", "n_selected", "alpha"],
        ascending=[False, True, False]
    )

    chosen = candidates.iloc[0]

    # Soft guard logic
    if chosen["stability_score"] < 0.45 and len(candidates) > 1:
        for _, row in candidates.iterrows():
            if row["stability_score"] > chosen["stability_score"] + 0.10 and row["n_selected"] <= chosen["n_selected"] + 3:
                chosen = row
                break

    final_sel = list(chosen["selected_union"])
    freqs = chosen["freqs"]

    # 7. Minimal zero-feature fallback using target_floor
    if len(final_sel) == 0:
        best_row = df.loc[best_idx]
        final_sel = list(best_row["selected_union"])
        if len(final_sel) == 0:
            final_sel = list(range(target_floor))

    # 6. Adaptive consolidation threshold logic
    consolidated = [i for i in final_sel if freqs[i] >= eff_sel_thresh]
    min_keep = max(target_floor, int(0.75 * len(final_sel)))
    if len(consolidated) >= min_keep:
        final_sel = consolidated

    # Backfill if over-sparse (below target_floor)
    was_backfilled = False
    if len(final_sel) < target_floor and n_features >= target_floor:
        was_backfilled = True
        needed = target_floor - len(final_sel)

        # Priority 1: features ordered by selection frequency descending
        avail = [i for i in range(n_features) if i not in final_sel]
        avail = sorted(avail, key=lambda i: freqs[i], reverse=True)
        final_sel.extend(avail[:needed])

    final_sel = sorted(set(final_sel))
    final_sel_idx = np.array(final_sel, dtype=int)
    final_names = [feature_names[i] for i in final_sel_idx]

    # Update full table for diagnostics
    df["normalized_n_selected"] = df["n_selected"] / max(1, n_features)
    df["adjusted_score"] = df["stability_score"] - sparsity_penalty_eff * df["normalized_n_selected"]

    return {
        "selected_idx": final_sel_idx,
        "selected_names": final_names,
        "best_alpha": df.loc[best_idx, "alpha"],
        "best_l1_ratio": df.loc[best_idx, "l1_ratio"],
        "chosen_alpha": chosen["alpha"],
        "chosen_l1_ratio": chosen["l1_ratio"],
        "best_score": best_score,
        "score_std": best_std,
        "threshold_score": thresh,
        "n_features_selected": len(final_sel_idx),
        "selection_frequency": freqs,
        "selected_feature_frequencies": freqs[final_sel_idx],
        "mean_jaccard": chosen["mean_jaccard"],
        "mean_sign_consistency": chosen.get("mean_sign_consistency"),
        "stability_score": chosen["stability_score"],
        "adjusted_score": chosen["adjusted_score"],
        "effective_selection_freq_threshold": eff_sel_thresh,
        "model_feature_floor": target_floor,
        "was_backfilled_to_floor": was_backfilled,
        "path_table": df
    }


def select_features_via_staged_en_rfe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps_train: Optional[np.ndarray],
    model_kind: str,
    feature_names: List[str],
    sample_weight_train: Optional[np.ndarray] = None,
    stage1_alpha_grid: Optional[np.ndarray] = None,
    stage1_l1_ratio_grid: Optional[List[float]] = None,
    stage2_alpha_grid: Optional[np.ndarray] = None,
    stage2_l1_ratio_grid: Optional[List[float]] = None,
    stage1_inner_n_splits: int = 3,
    stage2_inner_n_splits: int = 5,
    stage2_min_fold_hits: int = 3,
    first_cut_keep_frac: float = 0.60,
    first_cut_keep_k: Optional[int] = None,
    max_features_cap: Optional[int] = None,
    min_features_floor: int = 5,
    do_final_refine: bool = True,
    use_sign_consistency: bool = False,
) -> Dict:

    n_total_features = X_train.shape[1]

    # Defaults
    if stage1_alpha_grid is None:
        stage1_alpha_grid = np.logspace(-3, 0.5, 8)
    if stage1_l1_ratio_grid is None:
        stage1_l1_ratio_grid = [0.15, 0.40, 0.70]

    if stage2_alpha_grid is None:
        stage2_alpha_grid = np.logspace(-3, 0.25, 6)
    if stage2_l1_ratio_grid is None:
        stage2_l1_ratio_grid = [0.20, 0.50]

    # --- STAGE 1: PERFORMANCE-FIRST CUT ---
    res1 = select_features_via_elasticnet(
        X_train=X_train,
        y_train=y_train,
        timestamps_train=timestamps_train,
        model_kind=model_kind,
        feature_names=feature_names,
        alpha_grid=stage1_alpha_grid,
        l1_ratio_grid=stage1_l1_ratio_grid,
        sample_weight_train=sample_weight_train,
        inner_n_splits=stage1_inner_n_splits,
        max_features_cap=max_features_cap,
        min_features_floor=min_features_floor,
        use_sign_consistency=use_sign_consistency
    )

    sel1_idx = res1["selected_idx"]
    freqs1 = res1["selection_frequency"]

    if first_cut_keep_k is not None:
        target_k = first_cut_keep_k
    else:
        target_k = max(min_features_floor, int(np.ceil(first_cut_keep_frac * len(sel1_idx))))

    target_k = min(target_k, len(sel1_idx))

    # Rank Stage 1 features by frequency descending
    sel1_ranked = sorted(sel1_idx, key=lambda i: freqs1[i], reverse=True)
    first_cut_idx = np.array(sorted(sel1_ranked[:target_k]), dtype=int)

    # Floor safety
    first_cut_idx = _backfill_to_floor(first_cut_idx, freqs1, min_features_floor, n_total_features)

    # --- STAGE 2: 5-FOLD STABILITY PRUNING ---
    X2 = X_train[:, first_cut_idx]
    names2 = [feature_names[i] for i in first_cut_idx]

    res2 = select_features_via_elasticnet(
        X_train=X2,
        y_train=y_train,
        timestamps_train=timestamps_train,
        model_kind=model_kind,
        feature_names=names2,
        alpha_grid=stage2_alpha_grid,
        l1_ratio_grid=stage2_l1_ratio_grid,
        sample_weight_train=sample_weight_train,
        inner_n_splits=stage2_inner_n_splits,
        max_features_cap=max_features_cap,
        min_features_floor=min_features_floor,
        use_sign_consistency=use_sign_consistency
    )

    freqs2_local = res2["selection_frequency"]

    # Pruning mask logic
    stage2_keep_mask = freqs2_local >= (stage2_min_fold_hits / stage2_inner_n_splits)
    pruned_local = np.where(stage2_keep_mask)[0]

    # Floor safety
    pruned_local = _backfill_to_floor(pruned_local, freqs2_local, min_features_floor, len(names2))

    # Map back to global indices
    pruned_idx_global = first_cut_idx[pruned_local]

    # Decide if Stage 3 is necessary
    pct_retained = len(pruned_idx_global) / max(1, len(first_cut_idx))
    skip_stage3 = False
    if not do_final_refine or pct_retained >= 0.90:
        skip_stage3 = True

    # --- STAGE 3: OPTIONAL FINAL REFINEMENT ---
    if skip_stage3:
        final_global_idx = pruned_idx_global
        res3 = None
    else:
        X3 = X_train[:, pruned_idx_global]
        names3 = [feature_names[i] for i in pruned_idx_global]

        res3 = select_features_via_elasticnet(
            X_train=X3,
            y_train=y_train,
            timestamps_train=timestamps_train,
            model_kind=model_kind,
            feature_names=names3,
            alpha_grid=stage1_alpha_grid,
            l1_ratio_grid=stage1_l1_ratio_grid,
            sample_weight_train=sample_weight_train,
            inner_n_splits=stage1_inner_n_splits, # Use 3 folds again
            max_features_cap=max_features_cap,
            min_features_floor=min_features_floor,
            use_sign_consistency=use_sign_consistency
        )

        final_local = res3["selected_idx"]
        final_global_idx = pruned_idx_global[final_local]

    final_names = [feature_names[i] for i in final_global_idx]

    return {
        "selected_idx": final_global_idx,
        "selected_names": final_names,
        "stage1_result": res1,
        "stage2_result": res2,
        "stage3_result": res3,
        "first_cut_idx": first_cut_idx,
        "stage2_pruned_idx": pruned_idx_global,
        "n_features_stage1": len(res1["selected_idx"]),
        "n_features_first_cut": len(first_cut_idx),
        "n_features_stage2": len(pruned_idx_global),
        "n_features_final": len(final_global_idx),
    }
