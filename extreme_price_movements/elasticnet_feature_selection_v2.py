import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import ElasticNet
from scipy.stats import spearmanr
from extreme_price_movements.purged_cv import PurgedKFold

def _compute_inner_splits(timestamps: Optional[np.ndarray], n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    cv = PurgedKFold(n_splits=n_splits, purge=43200, embargo=43200, times=timestamps)
    dummy_X = np.empty((n_samples, 1))
    return list(cv.split(dummy_X))

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
        # Top-decile realized net
        k = max(1, int(len(y_p) * 0.10))
        idx = np.argpartition(y_p, -k)[-k:]
        return float(np.mean(y_t[idx]))

    elif model_kind == "downside":
        # Negative MAE (higher is better)
        return -float(np.mean(np.abs(y_p - y_t)))

    elif model_kind == "uncertainty":
        # Correlation
        corr, _ = spearmanr(y_p, y_t)
        return float(corr) if pd.notna(corr) else 0.0

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
    from extreme_price_movements.position_sizer_v2 import PredictionScaler
    fold_data = []
    for tr, va in splits:
        scaler = PredictionScaler()
        x_tr_s = scaler.fit_transform(X_train[tr])
        x_va_s = scaler.transform(X_train[va])
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

            for fd in fold_data:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
                try:
                    model.fit(fd["x_tr_s"], fd["y_tr"], sample_weight=fd["w_tr"])
                    preds = model.predict(fd["x_va_s"])
                    score = score_fold(fd["y_va"], preds, model_kind)
                    nonzero_idx = np.where(np.abs(model.coef_) > 1e-9)[0]
                except Exception:
                    score = -1e9
                    nonzero_idx = np.array([], dtype=int)

                fold_scores.append(score)
                fold_selected.append(nonzero_idx)
                if len(nonzero_idx) == 0:
                    all_zero_count += 1

            # Combine selected features across folds (union)
            union_selected = set()
            for sel in fold_selected:
                union_selected.update(sel)

            n_sel = len(union_selected)
            if n_sel > max_features_cap:
                continue # Skip dense candidates

            # Selection frequency
            freqs = np.zeros(n_features)
            for sel in fold_selected:
                freqs[sel] += 1
            freqs /= len(splits)

            mean_sel_freq = float(np.mean(freqs[list(union_selected)])) if union_selected else 0.0
            mean_jaccard = compute_pairwise_jaccard(fold_selected)

            stab_score = 0.7 * mean_sel_freq + 0.3 * mean_jaccard

            path_results.append({
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
                "n_selected": n_sel,
                "selected_union": list(union_selected),
                "mean_sel_freq": mean_sel_freq,
                "mean_jaccard": mean_jaccard,
                "stability_score": stab_score,
                "freqs": freqs
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

    # Optional guard: if sparsest has very bad stability (< 0.55), pick next
    chosen = candidates.iloc[0]
    for _, row in candidates.iterrows():
        if row["n_selected"] > 0 and row["stability_score"] >= 0.55:
            chosen = row
            break

    final_sel = chosen["selected_union"]
    if len(final_sel) == 0:
        # Fallback to full features if everything pruned to 0
        final_sel = list(range(n_features))

    # Optional final consolidation: intersection with freq >= threshold
    freqs = chosen["freqs"]
    consolidated = [i for i in final_sel if freqs[i] >= selection_freq_threshold]
    if len(consolidated) >= 1:
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
        "mean_sign_consistency": None,
        "stability_score": chosen["stability_score"],
        "path_table": df
    }
