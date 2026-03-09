import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import ElasticNet
from scipy.stats import spearmanr
from extreme_price_movements.purged_cv import PurgedKFold

def _compute_inner_splits(timestamps: Optional[np.ndarray], n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < max(20, n_splits * 5):
        mid = n_samples // 2
        return [(np.arange(0, mid), np.arange(mid, n_samples))]

    cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    splits = []
    for tr, va in cv.split(dummy_X):
        if len(tr) > 0 and len(va) > 0:
            splits.append((tr, va))

    if not splits:
        mid = n_samples // 2
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
    inner_n_splits: int = 2,
    max_features_cap: Optional[int] = None,
    stability_weight: float = 0.15,
    size_weight: float = 0.05,
    selection_freq_threshold: float = 0.60,
    use_sign_consistency: bool = False,
) -> Dict:

    n_samples, n_features = X_train.shape
    if max_features_cap is None:
        max_features_cap = n_features

    splits = _compute_inner_splits(timestamps_train, n_samples, inner_n_splits)
    if not splits:
        # Fallback block split
        mid = n_samples // 2
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
                    stab_score = 0.6 * mean_sel_freq + 0.25 * mean_jaccard + 0.15 * msc
                else:
                    stab_score = 0.7 * mean_sel_freq + 0.3 * mean_jaccard
            else:
                stab_score = 0.7 * mean_sel_freq + 0.3 * mean_jaccard

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
                "mean_sign_consistency": msc
            })

            # Prune path: if all folds selected 0 features, no need to try larger alphas for this l1_ratio
            if all_zero_count == len(splits):
                break

    if not path_results:
        return {
            "selected_idx": np.arange(n_features),
            "selected_names": feature_names,
            "best_alpha": 0.0, "best_l1_ratio": 0.0,
            "chosen_alpha": 0.0, "chosen_l1_ratio": 0.0,
            "best_score": 0.0, "score_std": 0.0, "threshold_score": 0.0,
            "n_features_selected": n_features,
            "selection_frequency": np.ones(n_features),
            "mean_jaccard": 1.0,
            "mean_sign_consistency": None,
            "stability_score": 1.0,
            "path_table": pd.DataFrame()
        }

    df = pd.DataFrame(path_results)

    # 1 std rule + stability
    best_idx = df["mean_score"].idxmax()
    best_score = df.loc[best_idx, "mean_score"]
    best_std = df.loc[best_idx, "std_score"]
    thresh = best_score - best_std

    candidates = df[df["mean_score"] >= thresh].copy()

    # Sort by: sparsest -> most stable -> largest alpha
    candidates = candidates.sort_values(
        by=["n_selected", "stability_score", "alpha"],
        ascending=[True, False, False]
    )

    chosen = candidates.iloc[0]
    for _, row in candidates.iterrows():
        if row["n_selected"] > 0 and row["stability_score"] >= 0.55:
            chosen = row
            break

    final_sel = chosen["selected_union"]

    # 7. Minimal zero-feature fallback
    if len(final_sel) == 0:
        best_row = df.loc[best_idx]
        fallback_sel = list(best_row["selected_union"])
        if len(fallback_sel) == 0:
            fallback_sel = list(range(min(n_features, 5)))
        final_sel = fallback_sel

    # 6. Gentler final consolidation
    freqs = chosen["freqs"]
    consolidated = [i for i in final_sel if freqs[i] >= selection_freq_threshold]
    min_keep = max(3, int(0.5 * len(final_sel)))
    if len(consolidated) >= min_keep:
        final_sel = consolidated

    final_sel_idx = np.array(sorted(final_sel), dtype=int)
    final_names = [feature_names[i] for i in final_sel_idx]

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
        "mean_jaccard": chosen["mean_jaccard"],
        "mean_sign_consistency": chosen["mean_sign_consistency"],
        "stability_score": chosen["stability_score"],
        "path_table": df
    }
