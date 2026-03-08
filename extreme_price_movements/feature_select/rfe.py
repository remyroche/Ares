import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Tuple, Literal, Optional, Dict, Any, Callable
from extreme_price_movements.feature_select.scoring import (
    UtilityConfig, FeatureSelectConfig, compute_utility, compute_bootstrap_ci, compute_composite_score
)
from extreme_price_movements.feature_select.importance import block_permutation_importance, compute_shap_importance
from extreme_price_movements.feature_select.cv import CVConfig, create_cv_splits
from extreme_price_movements.utils import tprint

def train_lgbm(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_X: pd.DataFrame,
    val_y: np.ndarray,
    model_kind: Literal["binary", "regression", "quantile"],
    quantile_alpha: Optional[float],
    lgbm_params: dict,
    seed: int
):
    """Trains a LightGBM model and returns the fitted model and val_pred."""
    params = lgbm_params.copy()
    params["random_state"] = seed
    params["n_jobs"] = 3
    params["verbose"] = -1

    n_estimators = params.pop("n_estimators", 1000)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    # Extract sample weights if present
    train_w = train_X['sample_weight'].values if 'sample_weight' in train_X.columns else None
    val_w = val_X['sample_weight'].values if 'sample_weight' in val_X.columns else None

    # Drop sample_weight column before training to prevent DataLeakage
    X_tr = train_X.drop(columns=['sample_weight']) if 'sample_weight' in train_X.columns else train_X
    X_va = val_X.drop(columns=['sample_weight']) if 'sample_weight' in val_X.columns else val_X

    # Handle evaluation metric
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    if model_kind == "binary":
        model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
        model.fit(X_tr, train_y, sample_weight=train_w, eval_set=[(X_va, val_y)], callbacks=callbacks)
        val_pred = model.predict_proba(X_va)[:, 1]
    elif model_kind == "regression":
        model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)
        model.fit(X_tr, train_y, sample_weight=train_w, eval_set=[(X_va, val_y)], callbacks=callbacks)
        val_pred = model.predict(X_va)
    elif model_kind == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = quantile_alpha
        model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)
        model.fit(X_tr, train_y, sample_weight=train_w, eval_set=[(X_va, val_y)], callbacks=callbacks)
        val_pred = model.predict(X_va)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    return model, val_pred

def run_rfe(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    time_index: Optional[pd.Series],
    model_kind: Literal["binary", "regression", "quantile"],
    quantile_alpha: Optional[float],
    cv_config: CVConfig,
    lgbm_params: dict,
    utility_config: UtilityConfig,
    fs_config: FeatureSelectConfig,
    random_seed: int = 42,
    max_samples: int = 8000,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Runs Recursive Feature Elimination with LightGBM."""
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
    if time_index is not None:
        time_index = np.asarray(time_index)

    if len(X) > max_samples:
        tprint(f"      [run_rfe] Subsampling from {len(X)} to {max_samples} for faster iterations")
        indices = np.linspace(0, len(X) - 1, max_samples, dtype=np.int32)
        X = X.iloc[indices].copy()
        y = y[indices]
        if groups is not None:
            groups = groups[indices]
        if time_index is not None:
            time_index = time_index[indices]

    current_features = [c for c in X.columns if c != "sample_weight" and c != "realized_utility"]
    min_features = max(1, min(int(fs_config.min_features), len(current_features)))
    max_features = fs_config.max_features
    if max_features is not None:
        max_features = max(min_features, min(int(max_features), len(current_features)))
    splits = create_cv_splits(X, cv_config, time_index)

    rfe_trace = []
    iteration = 0
    best_utility = -np.inf
    feature_scores = pd.DataFrame({
        "feature": current_features,
        "perm_importance_mean": np.nan,
        "perm_importance_std": np.nan,
        "shap_mean_abs": np.nan,
        "shap_presence": np.nan,
        "composite_score": np.nan,
        "rank": np.arange(1, len(current_features) + 1, dtype=int),
    })

    if len(current_features) <= min_features:
        tprint(
            f"      [run_rfe] Skipping RFE: {len(current_features)} features <= min_features={min_features}"
        )
        rfe_trace.append({
            "iter": 0,
            "n_features": len(current_features),
            "oos_utility_mean": float("nan"),
            "oos_utility_ci_low": float("nan"),
            "oos_utility_ci_high": float("nan"),
            "oos_metric_mean": float("nan"),
            "dropped_features": [],
            "skipped_rfe": True,
        })
        return current_features, feature_scores, pd.DataFrame(rfe_trace)

    while len(current_features) > min_features:
        iteration += 1
        tprint(f"      [run_rfe] Iteration {iteration}: {len(current_features)} features remaining")

        # 1. Run CV
        fold_utilities = []
        fold_metrics = []
        perm_results = []
        shap_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx][current_features].copy(), y[train_idx]
            X_va, y_va = X.iloc[val_idx][current_features].copy(), y[val_idx]

            # Downcast to float32
            for col in current_features:
                if X_tr[col].dtype == 'float64':
                    X_tr[col] = X_tr[col].astype('float32')
                if X_va[col].dtype == 'float64':
                    X_va[col] = X_va[col].astype('float32')

            # Add sample_weight back if present in original X
            if 'sample_weight' in X.columns:
                X_tr['sample_weight'] = X.iloc[train_idx]['sample_weight']
                X_va['sample_weight'] = X.iloc[val_idx]['sample_weight']

            # Train Model
            model, val_pred = train_lgbm(
                X_tr, y_tr, X_va, y_va,
                model_kind, quantile_alpha, lgbm_params, random_seed + fold_idx
            )

            # Compute Metric and Utility
            # In real implementation, metric_fn should be defined properly. Using dummy MSE/logloss for simplicity.
            if model_kind == "binary":
                from sklearn.metrics import log_loss
                base_metric = log_loss(y_va, val_pred)
                def metric_fn(y, p, X): return log_loss(y, p)
            else:
                from sklearn.metrics import mean_squared_error
                base_metric = mean_squared_error(y_va, val_pred)
                def metric_fn(y, p, X): return mean_squared_error(y, p)

            base_utility = compute_utility(y_va, val_pred, utility_config, X_va)

            def utility_fn(y, p, X): return compute_utility(y, p, utility_config, X)

            fold_utilities.append(base_utility)
            fold_metrics.append(base_metric)

            # Compute Importance
            block_ids = groups[val_idx] if groups is not None else None
            time_index_va = time_index[val_idx] if time_index is not None else None

            perm_df = block_permutation_importance(
                model, X_va, y_va, val_pred, base_metric, base_utility,
                current_features, metric_fn, utility_fn, block_ids,
                fs_config.n_repeats_perm, random_seed + fold_idx,
                max_samples=fs_config.perm_sample
            )
            perm_results.append(perm_df)

            shap_df = compute_shap_importance(
                model, X_va, current_features, model_kind, fs_config.shap_sample, random_seed + fold_idx
            )
            shap_results.append(shap_df)

        # Aggregate Importance
        mean_utility = np.mean(fold_utilities)
        # Check if we have standard deviation or if this is too small
        _, ci_low, ci_high = compute_bootstrap_ci(np.array(fold_utilities), n_boot=200, seed=random_seed)
        mean_metric = np.mean(fold_metrics)

        # Combine per-fold DataFrames
        all_perm = pd.concat(perm_results).groupby("feature").mean().reset_index()
        all_shap = pd.concat(shap_results).groupby("feature").mean().reset_index()

        # Calculate shap_presence
        shap_presence_counts = {f: 0 for f in current_features}
        for shap_df in shap_results:
            top_features = shap_df.sort_values("shap_mean_abs", ascending=False).head(fs_config.topk_presence)["feature"]
            for f in top_features:
                shap_presence_counts[f] += 1

        all_shap["shap_presence"] = all_shap["feature"].map(shap_presence_counts) / len(splits)

        feature_scores = pd.merge(all_shap, all_perm, on="feature")

        # Compute Composite Score
        composite_scores = compute_composite_score(
            feature_scores["perm_importance_mean"].values,
            feature_scores["perm_importance_std"].values,
            feature_scores["shap_mean_abs"].values,
            np.zeros(len(feature_scores)), # Placeholder for SHAP std
            feature_scores["shap_presence"].values,
            fs_config
        )

        feature_scores["composite_score"] = composite_scores
        feature_scores = feature_scores.sort_values("composite_score", ascending=False)
        feature_scores["rank"] = np.arange(1, len(feature_scores) + 1)

        # Log trace
        rfe_trace.append({
            "iter": iteration,
            "n_features": len(current_features),
            "oos_utility_mean": mean_utility,
            "oos_utility_ci_low": ci_low,
            "oos_utility_ci_high": ci_high,
            "oos_metric_mean": mean_metric,
        })
        tprint(f"      [run_rfe]   OOS Utility: {mean_utility:.6f} (+/- {ci_high - mean_utility:.6f})")

        # Early stopping or utility check
        if iteration == 1:
            best_utility = mean_utility
        else:
            if mean_utility < best_utility - fs_config.utility_drop_tol:
                # Revert to previous feature set (this implementation proceeds and just notes it)
                break
            else:
                best_utility = max(best_utility, mean_utility)

        # Determine features to drop
        n_drop = max(2, int(0.15 * (len(current_features) - min_features)))
        n_drop = min(n_drop, len(current_features) - min_features)

        if n_drop <= 0:
            break

        features_to_drop = feature_scores.tail(n_drop)["feature"].tolist()
        current_features = [f for f in current_features if f not in features_to_drop]
        rfe_trace[-1]["dropped_features"] = features_to_drop

    if max_features is not None and len(current_features) > int(max_features):
        feature_scores = feature_scores[feature_scores["feature"].isin(current_features)].copy()
        feature_scores = feature_scores.sort_values("composite_score", ascending=False)
        keep_n = max(min_features, int(max_features))
        current_features = feature_scores.head(keep_n)["feature"].tolist()
        rfe_trace.append({
            "iter": iteration + 1,
            "n_features": len(current_features),
            "oos_utility_mean": float(best_utility) if np.isfinite(best_utility) else 0.0,
            "oos_utility_ci_low": float("nan"),
            "oos_utility_ci_high": float("nan"),
            "oos_metric_mean": float("nan"),
            "dropped_features": [],
            "hard_cap_applied": True,
            "max_features_cap": int(max_features),
        })

    return current_features, feature_scores, pd.DataFrame(rfe_trace)
