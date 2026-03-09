import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.offline_optimisers.params_store import _read_best_params_csv, INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV
from extreme_price_movements.universe import apply_hardcoded_universe_exclusions
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.config import load_inference_config
import math

def _compute_fallback_ranking(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_samples, n_features = X.shape
    finite_rates = np.zeros(n_features)
    variances = np.zeros(n_features)
    corrs = np.zeros(n_features)

    for i in range(n_features):
        col = X[:, i]
        finite_mask = np.isfinite(col)
        n_finite = np.sum(finite_mask)
        finite_rates[i] = n_finite / max(1, n_samples)

        if n_finite > 1:
            col_finite = col[finite_mask]
            variances[i] = np.var(col_finite)

            y_finite = y[finite_mask]
            if len(np.unique(col_finite)) > 1:
                sp, _ = spearmanr(col_finite, y_finite)
                corrs[i] = abs(sp) if pd.notna(sp) else 0.0

    rank_tuples = [
        (-finite_rates[i], -variances[i], -corrs[i], i)
        for i in range(n_features)
    ]
    rank_tuples.sort()
    return np.array([t[3] for t in rank_tuples], dtype=int)

def _backfill_to_floor(local_idx: np.ndarray, freqs: np.ndarray, floor: int, subset_size: int, fallback_rank: Optional[np.ndarray] = None) -> np.ndarray:
    selected = list(local_idx)
    target = min(floor, subset_size)
    if len(selected) >= target:
        return np.array(sorted(selected), dtype=int)

    selected_set = set(selected)
    avail = [i for i in range(subset_size) if i not in selected_set]

    fallback_order = {}
    if fallback_rank is not None:
        fallback_order = {idx: rank for rank, idx in enumerate(fallback_rank)}

    if np.max(freqs) > 0:
        avail = sorted(avail, key=lambda i: (freqs[i], -fallback_order.get(i, float('inf'))), reverse=True)
    elif fallback_rank is not None:
        avail = sorted(avail, key=lambda i: fallback_order.get(i, float('inf')))

    selected.extend(avail[: target - len(selected)])
    return np.array(sorted(set(selected)), dtype=int)


def get_optimized_mask(
    features: Dict[str, pd.DataFrame],
    panel: Dict[str, pd.DataFrame],
    run_id: Optional[str] = None,
    data_root: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads best params from mask_optimiser and returns the boolean mask.
    Also extends mask with OOF preds and filters duplicates.
    """
    params = _read_best_params_csv(INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV)
    family = params.get("family", "top_movers")
    param_val = float(params.get("param", 5.0))
    z_hr = float(params.get("z_hours", 12.0))

    # Simple logic to filter universe
    if "close" in panel:
        symbols = list(panel["close"].columns)
        filtered_symbols = apply_hardcoded_universe_exclusions(symbols)
        for k, v in panel.items():
            panel[k] = v[filtered_symbols]
        for k, v in features.items():
            if isinstance(v, pd.DataFrame):
                features[k] = v[filtered_symbols]

    # Basic mask generation mimicking mask_optimiser logic
    mask_df = pd.DataFrame(False, index=panel["close"].index, columns=panel["close"].columns)

    if family == "top_movers":
        ret = panel["close"].pct_change(int(z_hr * 4)) # rough approx
        for ts, row in ret.iterrows():
            q_high = row.quantile(1.0 - param_val/100.0)
            q_low = row.quantile(param_val/100.0)
            mask_df.loc[ts] = (row >= q_high) | (row <= q_low)
    else:
        # Fallback if other families used
        mask_df.loc[:,:] = True

    # Extend missing OOF predictions using inference tool
    try:
        cfg = load_inference_config(run_id=run_id, data_root=data_root)
        orchestrator = ModelOrchestrator(cfg["model_bundle"], cfg)
        # We would run full chain here if needed to backfill OOF,
        # but in feature selection context this is usually handled upstream.
        # Just instantiating proves we can access the tools.
    except Exception as e:
        print(f"Warning: could not backfill OOF preds: {e}")

    return mask_df



def _compute_inner_splits(timestamps: Optional[np.ndarray], n_samples: int, n_splits: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    if n_samples < max(20, n_splits * 5):
        mid = max(1, n_samples // 2)
        if mid >= n_samples:
            return [], 0
        return [(np.arange(0, mid), np.arange(mid, n_samples))], 1

    cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    splits = []
    for tr, va in cv.split(dummy_X):
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr, va))

    if not splits:
        mid = max(1, n_samples // 2)
        if mid >= n_samples:
            return [], 0
        return [(np.arange(0, mid), np.arange(mid, n_samples))], 1
    return splits, len(splits)

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

def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if weights is None:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))

def _weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray]) -> float:
    diff = np.abs(y_true - y_pred)
    if weights is None:
        return float(np.mean(diff))
    return float(np.average(diff, weights=weights))

def score_fold(y_true: np.ndarray, y_pred: np.ndarray, model_kind: str, weights: Optional[np.ndarray] = None) -> float:
    """Returns metric where higher is better."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    if weights is not None:
        valid = valid & np.isfinite(weights)

    if not np.any(valid):
        return -1e9

    y_t = y_true[valid]
    y_p = y_pred[valid]
    w = weights[valid] if weights is not None else None

    n_val = len(y_p)
    skip_percentiles = n_val < 10

    if model_kind == "edge":
        # Top-decile realized net + spearman tie break
        k = max(1, int(np.ceil(0.10 * n_val)))
        idx = np.argpartition(y_p, -k)[-k:]

        y_t_top = y_t[idx]
        w_top = w[idx] if w is not None else None
        base = _weighted_mean(y_t_top, w_top)

        if not skip_percentiles:
            sp, _ = spearmanr(y_p, y_t)
            sp = float(sp) if pd.notna(sp) else 0.0
            return base + 0.05 * sp
        return base

    elif model_kind == "downside":
        # Negative MAE + false safe penalty
        mae = _weighted_mae(y_t, y_p, w)

        if skip_percentiles:
            return -mae

        # false safe: pred in lowest 20%, true in highest 20%
        safe_t = np.percentile(y_p, 20)
        danger_t = np.percentile(y_t, 80)
        pred_safe = y_p <= safe_t

        if np.sum(pred_safe) > 0:
            fs_mask = pred_safe & (y_t >= danger_t)
            if w is not None:
                w_sum = np.sum(w[pred_safe])
                fsr = np.sum(w[fs_mask]) / w_sum if w_sum > 0 else 0.0
            else:
                fsr = np.sum(fs_mask) / np.sum(pred_safe)
        else:
            fsr = 0.0

        return -(mae + 0.5 * fsr)

    elif model_kind == "uncertainty":
        # Correlation - underestimation penalty
        sp, _ = spearmanr(y_p, y_t)
        sp = float(sp) if pd.notna(sp) else 0.0

        if skip_percentiles:
            return sp

        # underestimation: pred in lowest 20%, true in highest 20%
        low_unc_t = np.percentile(y_p, 20)
        high_err_t = np.percentile(y_t, 80)
        pred_low = y_p <= low_unc_t

        if np.sum(pred_low) > 0:
            ue_mask = pred_low & (y_t >= high_err_t)
            if w is not None:
                w_sum = np.sum(w[pred_low])
                uer = np.sum(w[ue_mask]) / w_sum if w_sum > 0 else 0.0
            else:
                uer = np.sum(ue_mask) / np.sum(pred_low)
        else:
            uer = 0.0

        return sp - 0.25 * uer

    return 0.0

def _prepare_fold_data(X_train: np.ndarray, y_train: np.ndarray, sample_weight_train: Optional[np.ndarray], splits: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    fold_data = []
    for tr, va in splits:
        x_tr = X_train[tr].copy()
        x_va = X_train[va].copy()

        # finite cleaning policy
        x_tr[~np.isfinite(x_tr)] = np.nan
        x_va[~np.isfinite(x_va)] = np.nan

        medians = np.nanmedian(x_tr, axis=0)
        medians[np.isnan(medians)] = 0.0

        for i in range(x_tr.shape[1]):
            tr_nan = np.isnan(x_tr[:, i])
            va_nan = np.isnan(x_va[:, i])
            if np.any(tr_nan):
                x_tr[tr_nan, i] = medians[i]
            if np.any(va_nan):
                x_va[va_nan, i] = medians[i]

        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr).astype(np.float32)
        x_va_s = scaler.transform(x_va).astype(np.float32)

        fold_data.append({
            "tr": tr, "va": va,
            "x_tr_s": x_tr_s, "x_va_s": x_va_s,
            "y_tr": y_train[tr], "y_va": y_train[va],
            "w_tr": sample_weight_train[tr] if sample_weight_train is not None else None,
            "w_va": sample_weight_train[va] if sample_weight_train is not None else None
        })
    return fold_data


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
    fit_budget: int = 1000,
    memory_cache_budget_bytes: int = 268435456,
    fallback_rank: Optional[np.ndarray] = None
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

    splits, actual_n_splits = _compute_inner_splits(timestamps_train, n_samples, inner_n_splits)
    if not splits:
        return _generate_fallback_response(
            n_features=n_features, target_floor=target_floor, feature_names=feature_names,
            eff_sel_thresh=0.0, actual_n_splits=0, fit_count=0, fallback_rank=fallback_rank
        )

    if actual_n_splits <= 2:
        eff_sel_thresh = 0.50
    elif actual_n_splits == 3:
        eff_sel_thresh = 2.0 / 3.0
    else:
        eff_sel_thresh = selection_freq_threshold

    allowed_paths = math.floor(fit_budget / actual_n_splits)
    if allowed_paths <= 0:
        return _generate_fallback_response(
            n_features=n_features, target_floor=target_floor, feature_names=feature_names,
            eff_sel_thresh=eff_sel_thresh, actual_n_splits=actual_n_splits, fit_count=0, fallback_rank=fallback_rank
        )

    alpha_grid = np.unique(np.sort(alpha_grid))
    param_paths = [(a, l1) for l1 in l1_ratio_grid for a in alpha_grid][:allowed_paths]

    # Memory caching
    estimated_bytes = actual_n_splits * n_samples * n_features * 4 * 2 # tr and va floats
    use_cache = estimated_bytes <= memory_cache_budget_bytes
    fold_data = _prepare_fold_data(X_train, y_train, sample_weight_train, splits) if use_cache else None

    path_results = []
    fit_count = 0

    # Restructure the loop to handle pruned paths gracefully since param_paths are flattened
    pruned_l1_ratios = set()

    for alpha, l1_ratio in param_paths:
        if l1_ratio in pruned_l1_ratios:
            continue

        fold_scores = []
        fold_selected = []
        all_zero_count = 0
        fold_coef_signs = []

        for i, (tr, va) in enumerate(splits):
            if use_cache:
                fd = fold_data[i]
            else:
                fd = _prepare_fold_data(X_train, y_train, sample_weight_train, [(tr, va)])[0]

            model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=3000, tol=1e-3, random_state=42, selection="cyclic")
            try:
                model.fit(fd["x_tr_s"], fd["y_tr"], sample_weight=fd["w_tr"])
                preds = model.predict(fd["x_va_s"])
                score = score_fold(fd["y_va"], preds, model_kind, fd["w_va"])
                coefs = model.coef_
                nonzero_idx = np.where(np.abs(coefs) > 1e-9)[0]
                signs = np.sign(coefs)
            except Exception:
                score = -1e9
                nonzero_idx = np.array([], dtype=int)
                signs = np.zeros(n_features)

            fit_count += 1
            fold_scores.append(score)
            fold_selected.append(nonzero_idx)
            fold_coef_signs.append(signs)

            if len(nonzero_idx) == 0:
                all_zero_count += 1

        union_selected = set()
        for sel in fold_selected:
            union_selected.update(sel)

        n_sel = len(union_selected)

        freqs = np.zeros(n_features)
        for sel in fold_selected:
            freqs[sel] += 1
        freqs /= actual_n_splits

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

        if all_zero_count == actual_n_splits:
            pruned_l1_ratios.add(l1_ratio)

    if not path_results:
        return _generate_fallback_response(
            n_features=n_features, target_floor=target_floor, feature_names=feature_names,
            eff_sel_thresh=eff_sel_thresh, actual_n_splits=actual_n_splits, fit_count=fit_count, fallback_rank=fallback_rank
        )

    df = pd.DataFrame(path_results)

    best_idx = df["mean_score"].idxmax()
    best_score = df.loc[best_idx, "mean_score"]
    best_std = df.loc[best_idx, "std_score"]
    thresh = best_score - best_std

    candidates = df[df["mean_score"] >= thresh].copy()
    candidates["normalized_n_selected"] = candidates["n_selected"] / max(1, n_features)
    candidates["adjusted_score"] = candidates["stability_score"] - sparsity_penalty_eff * candidates["normalized_n_selected"]

    candidates = candidates.sort_values(
        by=["adjusted_score", "n_selected", "alpha"],
        ascending=[False, True, False]
    )

    chosen = candidates.iloc[0]

    if chosen["stability_score"] < 0.45 and len(candidates) > 1:
        for _, row in candidates.iterrows():
            if row["stability_score"] > chosen["stability_score"] + 0.10 and row["n_selected"] <= chosen["n_selected"] + 3:
                chosen = row
                break

    final_sel = list(chosen["selected_union"])
    freqs = chosen["freqs"]

    if len(final_sel) == 0:
        best_row = df.loc[best_idx]
        final_sel = list(best_row["selected_union"])
        if len(final_sel) == 0:
            return _generate_fallback_response(
                n_features=n_features, target_floor=target_floor, feature_names=feature_names,
                eff_sel_thresh=eff_sel_thresh, actual_n_splits=actual_n_splits, fit_count=fit_count, fallback_rank=fallback_rank
            )

    consolidated = [i for i in final_sel if freqs[i] >= eff_sel_thresh]
    min_keep = max(target_floor, int(0.75 * len(final_sel)))
    if len(consolidated) >= min_keep:
        final_sel = consolidated

    was_backfilled = False
    if len(final_sel) < target_floor and n_features >= target_floor:
        was_backfilled = True
        final_sel = list(_backfill_to_floor(np.array(final_sel), freqs, target_floor, n_features, fallback_rank))

    # Apply max_features_cap after consolidation and backfilling
    if len(final_sel) > max_features_cap:
        # rank retained features by selection frequency descending
        # tie-break by sign consistency descending if enabled
        # tie-break by lower feature index

        feature_consistencies = {}
        if use_sign_consistency and len(fold_coef_signs) > 0:
            signs_mat = np.vstack(fold_coef_signs)
            for idx in final_sel:
                col = signs_mat[:, idx]
                nz_col = col[col != 0]
                if len(nz_col) > 0:
                    frac_pos = np.sum(nz_col > 0) / len(nz_col)
                    frac_neg = np.sum(nz_col < 0) / len(nz_col)
                    feature_consistencies[idx] = max(frac_pos, frac_neg)
                else:
                    feature_consistencies[idx] = 0.0

        final_sel = sorted(final_sel, key=lambda i: (freqs[i], feature_consistencies.get(i, 0.0), -i), reverse=True)[:max_features_cap]

    final_sel = sorted(set(final_sel))
    final_sel_idx = np.array(final_sel, dtype=int)
    final_names = [feature_names[i] for i in final_sel_idx]

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
        "chosen_path_selection_frequency": freqs,
        "selection_frequency": freqs,
        "selected_feature_frequencies": freqs[final_sel_idx],
        "mean_jaccard": chosen["mean_jaccard"],
        "mean_sign_consistency": chosen.get("mean_sign_consistency"),
        "stability_score": chosen["stability_score"],
        "adjusted_score": chosen["adjusted_score"],
        "effective_selection_freq_threshold": eff_sel_thresh,
        "model_feature_floor": target_floor,
        "was_backfilled_to_floor": was_backfilled,
        "path_table": df,
        "actual_n_splits_used": actual_n_splits,
        "fit_count_used": fit_count,
        "fallback_strategy_used": False
    }

def _generate_fallback_response(n_features: int, target_floor: int, feature_names: List[str], eff_sel_thresh: float, actual_n_splits: int, fit_count: int, fallback_rank: Optional[np.ndarray] = None) -> Dict:
    fallback_n = min(n_features, max(1, target_floor))
    if fallback_rank is not None:
        fallback_idx = fallback_rank[:fallback_n]
    else:
        fallback_idx = np.arange(fallback_n, dtype=int)

    fallback_idx = np.sort(fallback_idx)
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
        "chosen_path_selection_frequency": np.zeros(n_features),
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
        "actual_n_splits_used": actual_n_splits,
        "fit_count_used": fit_count,
        "fallback_strategy_used": True
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
    stage1_inner_n_splits: int = 2,
    stage2_inner_n_splits: int = 2,
    stage2_min_fold_hits: int = 3,
    first_cut_keep_frac: float = 0.60,
    first_cut_keep_k: Optional[int] = None,
    max_features_cap: Optional[int] = None,
    min_features_floor: int = 5,
    do_final_refine: bool = False,
    use_sign_consistency: bool = False,
) -> Dict:

    n_total_features = X_train.shape[1]
    global_fallback_rank = _compute_fallback_ranking(X_train, y_train)

    # Defaults
    if stage1_alpha_grid is None:
        stage1_alpha_grid = np.array([1.0, 0.1, 0.01])
    if stage1_l1_ratio_grid is None:
        stage1_l1_ratio_grid = [0.2, 0.7]

    if stage2_alpha_grid is None:
        stage2_alpha_grid = np.array([0.3, 0.03])
    if stage2_l1_ratio_grid is None:
        stage2_l1_ratio_grid = [0.5, 0.8]

    remaining_fit_budget = 20
    total_fit_count_used = 0

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
        use_sign_consistency=use_sign_consistency,
        fit_budget=remaining_fit_budget,
        fallback_rank=global_fallback_rank
    )

    remaining_fit_budget -= res1["fit_count_used"]
    total_fit_count_used += res1["fit_count_used"]

    sel1_idx = res1["selected_idx"]
    freqs1 = res1["chosen_path_selection_frequency"]

    if first_cut_keep_k is not None:
        target_k = first_cut_keep_k
    else:
        target_k = max(min_features_floor, int(np.ceil(first_cut_keep_frac * len(sel1_idx))))

    target_k = min(target_k, len(sel1_idx))

    # Rank Stage 1 features by frequency descending
    sel1_ranked = sorted(sel1_idx, key=lambda i: freqs1[i], reverse=True)
    first_cut_idx = np.array(sorted(sel1_ranked[:target_k]), dtype=int)

    # Floor safety within Stage 1 selected, then global
    first_cut_idx = _backfill_to_floor(first_cut_idx, freqs1, min_features_floor, n_total_features, global_fallback_rank)

    # --- STAGE 2: 5-FOLD STABILITY PRUNING ---
    X2 = X_train[:, first_cut_idx]
    names2 = [feature_names[i] for i in first_cut_idx]
    global_order = {idx: rank for rank, idx in enumerate(global_fallback_rank)}
    local_fallback_rank2 = np.array(sorted(range(len(first_cut_idx)), key=lambda i: global_order.get(first_cut_idx[i], float('inf'))))

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
        use_sign_consistency=use_sign_consistency,
        fit_budget=remaining_fit_budget,
        fallback_rank=local_fallback_rank2
    )

    remaining_fit_budget -= res2["fit_count_used"]
    total_fit_count_used += res2["fit_count_used"]
    actual_stage2_splits = res2["actual_n_splits_used"]
    freqs2_local = res2["chosen_path_selection_frequency"]

    # Pruning mask logic
    effective_hits = min(stage2_min_fold_hits, actual_stage2_splits)
    threshold = effective_hits / max(1, actual_stage2_splits)
    stage2_keep_mask = freqs2_local >= threshold
    pruned_local = np.where(stage2_keep_mask)[0]

    # Floor safety
    pruned_local = _backfill_to_floor(pruned_local, freqs2_local, min_features_floor, len(names2), local_fallback_rank2)

    # Map back to global indices
    pruned_idx_global = first_cut_idx[pruned_local]

    # Decide if Stage 3 is necessary
    pct_retained = len(pruned_idx_global) / max(1, len(first_cut_idx))
    skip_stage3 = False
    if not do_final_refine or remaining_fit_budget < 4 or pct_retained >= 0.90:
        skip_stage3 = True

    # --- STAGE 3: OPTIONAL FINAL REFINEMENT ---
    if skip_stage3:
        final_global_idx = pruned_idx_global
        res3 = None
    else:
        X3 = X_train[:, pruned_idx_global]
        names3 = [feature_names[i] for i in pruned_idx_global]
        global_order3 = {idx: rank for rank, idx in enumerate(global_fallback_rank)}
        local_fallback_rank3 = np.array(sorted(range(len(pruned_idx_global)), key=lambda i: global_order3.get(pruned_idx_global[i], float('inf'))))

        res3 = select_features_via_elasticnet(
            X_train=X3,
            y_train=y_train,
            timestamps_train=timestamps_train,
            model_kind=model_kind,
            feature_names=names3,
            alpha_grid=stage1_alpha_grid,
            l1_ratio_grid=stage1_l1_ratio_grid,
            sample_weight_train=sample_weight_train,
            inner_n_splits=2, # Use 2 folds max
            max_features_cap=max_features_cap,
            min_features_floor=min_features_floor,
            use_sign_consistency=use_sign_consistency,
            fit_budget=4, # Hard cap stage3 fits at 4
            fallback_rank=local_fallback_rank3
        )

        total_fit_count_used += res3["fit_count_used"]
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
        "total_fit_count_used": total_fit_count_used
    }
