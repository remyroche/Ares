import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from extreme_price_movements.utils import tprint
from extreme_price_movements.purged_cv import PurgedKFold

class ScaledLogisticRegression(LogisticRegression):
    """
    Wrapper to apply StandardScaler internally, ensuring sample_weight 
    is correctly passed to fit (bypassing Pipeline limitations with CalibratedClassifierCV).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler = StandardScaler()

    def fit(self, X, y, sample_weight=None):
        X_scaled = self.scaler.fit_transform(X)
        return super().fit(X_scaled, y, sample_weight=sample_weight)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict_proba(X_scaled)

def calculate_selection_score(y_true, y_prob, y_returns):
    """
    S_model = (AUC - 0.5) * BSS * IC
    Optimized:
    - Brier reference calculated analytically (O(1) from mean).
    - Spearman correlation via rank transformation + Pearson (O(N)).
    """
    # 1. AUC (Ranking)
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    # 2. Brier Skill Score (Calibration)
    # BS = mean((p - y)^2)
    # Reference: p = mean(y) constant
    # BS_ref = mean((mean(y) - y)^2) = var(y)
    bs_actual = brier_score_loss(y_true, y_prob)
    mean_val = np.mean(y_true)
    bs_ref = np.mean((y_true - mean_val) ** 2)

    bss = 1.0 - (bs_actual / (bs_ref + 1e-12))

    # 3. Information Coefficient (Magnitude)
    # Spearman correlation between predicted prob and ACTUAL CONTINUOUS RETURN
    # Optimized: rankdata + corrcoef
    # Handle constant inputs to avoid NaNs
    if np.std(y_prob) < 1e-9 or np.std(y_returns) < 1e-9:
        ic = 0.0
    else:
        rank_prob = rankdata(y_prob)
        rank_ret = rankdata(y_returns)
        # np.corrcoef returns 2x2 matrix
        ic = np.corrcoef(rank_prob, rank_ret)[0, 1]
        if np.isnan(ic):
            ic = 0.0

    # Combined Score: weighted sum (more forgiving than triple product)
    # AUC is primary (0-1 range centered at 0.5), IC adds magnitude signal
    auc_contrib = max(0, auc - 0.5)  # 0 to 0.5
    ic_contrib = max(0, ic)           # 0 to 1
    bss_contrib = max(0, bss)         # 0 to 1
    selection_score = 0.6 * auc_contrib + 0.3 * ic_contrib + 0.1 * bss_contrib

    return {
        "Selection_Score": selection_score,
        "AUC": auc,
        "BSS": bss,
        "IC": ic,
        "Brier": bs_actual
    }

class ModelRace(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="mr", n_splits=3):
        tprint(f"Entering function: __init__ in model_race.py")
        self.kind = kind
        self.n_splits = n_splits
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.race_sample_frac = 0.25
        self.race_early_stopping_rounds = 100

    @staticmethod
    def _compute_pos_weight(y):
        y_arr = np.asarray(y)
        pos = np.sum(y_arr == 1)
        neg = np.sum(y_arr == 0)
        if pos <= 0:
            return 1.0
        return max(1.0, float(neg) / float(pos))

    @staticmethod
    def _subsample_indices(indices, frac, seed):
        if frac >= 1.0 or len(indices) <= 1:
            return indices
        target_size = max(1, int(np.ceil(len(indices) * frac)))
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(indices, size=target_size, replace=False))
        return picked

    def _fit_model(self, model, X_tr, y_tr, X_val=None, y_val=None, sample_weight=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        pos_weight = self._compute_pos_weight(y_tr)

        if isinstance(model, ScaledLogisticRegression):
            model.set_params(class_weight={0: 1.0, 1: pos_weight})
        elif isinstance(model, ExtraTreesClassifier):
            model.set_params(class_weight={0: 1.0, 1: pos_weight})

        if isinstance(model, CatBoostClassifier):
            model.set_params(scale_pos_weight=pos_weight)
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": (X_val, y_val),
                    "early_stopping_rounds": self.race_early_stopping_rounds,
                    "use_best_model": True,
                })
        elif isinstance(model, XGBClassifier):
            model.set_params(scale_pos_weight=pos_weight, eval_metric="auc")
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                    "early_stopping_rounds": self.race_early_stopping_rounds,
                })
        elif isinstance(model, LGBMClassifier):
            model.set_params(scale_pos_weight=pos_weight)
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": [(X_val, y_val)],
                    "eval_metric": "auc",
                    "callbacks": [],
                })
                try:
                    from lightgbm import early_stopping
                    fit_kwargs["callbacks"].append(early_stopping(self.race_early_stopping_rounds, verbose=False))
                except Exception:
                    pass

        model.fit(X_tr, y_tr, **fit_kwargs)

    def _get_candidates(self, race_mode=True):
        tprint(f"Entering function: _get_candidates in model_race.py (race_mode={race_mode})")
        candidates = {}

        # Scaling factors for race vs final
        n_est_et = 120 if race_mode else 1000
        n_iter_cb = 180 if race_mode else 1000
        n_est_xgb = 1 if race_mode else 10 # 1*150=150 vs 10*150=1500 trees
        n_est_lgbm = 250 if race_mode else 1000

        # 1. Baseline
        # ScaledLogisticRegression (Solves Pipeline+SampleWeight issue)
        candidates["elasticnet"] = ScaledLogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=2000, random_state=42
        )

        # 2. ExtraTrees
        candidates["extratrees"] = ExtraTreesClassifier(
            n_estimators=n_est_et,
            max_depth=7,
            min_samples_leaf=50,
            max_features=0.5,
            bootstrap=True,
            criterion='entropy',
            n_jobs=-1,
            random_state=42
        )

        # 3. CatBoost
        candidates["catboost"] = CatBoostClassifier(
            iterations=n_iter_cb,
            learning_rate=0.02,
            l2_leaf_reg=20,
            depth=5,
            subsample=0.6,
            colsample_bylevel=0.6,
            random_strength=10,
            min_data_in_leaf=50,
            verbose=0,
            loss_function='Logloss',
            eval_metric='AUC',
            allow_writing_files=False,
            random_state=42
        )

        # 4. XGBoost
        candidates["xgboost"] = XGBClassifier(
            n_estimators=n_est_xgb,
            num_parallel_tree=150,
            max_depth=6,
            gamma=4,
            reg_lambda=5,
            reg_alpha=1,
            subsample=0.6,
            colsample_bylevel=0.6,
            max_delta_step=3,
            n_jobs=-1,
            random_state=42
        )

        # 5. LightGBM
        candidates["lightgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=n_est_lgbm,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=100,
            subsample=0.6,
            colsample_bytree=0.6,
            subsample_freq=1,
            reg_alpha=1.0,
            reg_lambda=20.0,
            min_split_gain=0.02,
            max_bin=127,
            min_data_in_bin=25,
            n_jobs=-1,
            verbosity=-1,
            random_state=42
        )

        return candidates

    def fit(self, X, y, sample_weight=None, returns=None):
        """
        X: features
        y: binary target
        sample_weight: weights for training
        returns: continuous returns for IC calculation (validation)
        """
        tprint(f"Entering function: fit in model_race.py")
        self.oof_probs = None  # Will store OOF predictions from best model

        # 0. Preparation
        if returns is None:
            returns = y

        # Optimize: Convert to numpy once if possible (and suitable for all models)
        # ExtraTrees/XGBoost prefer numpy. CatBoost handles both but numpy is fine if no categorical features.
        # We assume numeric features here.
        X_np = X
        use_numpy = False
        if hasattr(X, "iloc"):
            try:
                # Use float32 to save memory and match FastFuncs usage
                X_np = X.to_numpy(dtype=np.float32, copy=False)
                use_numpy = True
            except (ValueError, TypeError):
                # Fallback if conversion fails (e.g. mixed types)
                use_numpy = False

        # Cache CV splits
        tscv = PurgedKFold(n_splits=self.n_splits, purge=5, embargo=2)
        cached_splits = list(tscv.split(X))

        # 1. The Race
        candidates = self._get_candidates(race_mode=True)
        results = {}

        def safe_slice(arr, idx):
            if hasattr(arr, "iloc"): return arr.iloc[idx]
            return arr[idx]

        # Store per-model detailed metrics for reporting
        detailed_metrics = {}

        for name, model in candidates.items():
            tprint(f"Race: Training {name}...")
            fold_scores = []
            fold_aucs = []
            fold_ics = []
            fold_bss = []
            fold_logloss = []
            fold_accuracy = []

            try:
                for fold_i, (train_idx, val_idx) in enumerate(cached_splits):
                    race_train_idx = self._subsample_indices(train_idx, self.race_sample_frac, seed=42 + fold_i)
                    race_val_idx = self._subsample_indices(val_idx, self.race_sample_frac, seed=142 + fold_i)

                    if use_numpy:
                        X_tr, X_val = X_np[race_train_idx], X_np[race_val_idx]
                    else:
                        X_tr, X_val = X.iloc[race_train_idx], X.iloc[race_val_idx]

                    y_tr = safe_slice(y, race_train_idx)
                    y_val = safe_slice(y, race_val_idx)
                    w_tr = safe_slice(sample_weight, race_train_idx) if sample_weight is not None else None
                    ret_val = safe_slice(returns, race_val_idx)

                    self._fit_model(model, X_tr, y_tr, X_val=X_val, y_val=y_val, sample_weight=w_tr)
                    probs = model.predict_proba(X_val)[:, 1]

                    metrics = calculate_selection_score(y_val, probs, ret_val)
                    fold_scores.append(metrics["Selection_Score"])
                    fold_aucs.append(metrics["AUC"])
                    fold_ics.append(metrics["IC"])
                    fold_bss.append(metrics["BSS"])

                    try:
                        fold_logloss.append(log_loss(y_val, probs))
                    except:
                        fold_logloss.append(np.nan)
                    fold_accuracy.append(accuracy_score(y_val, probs > 0.5))

                avg_score = np.nanmean(fold_scores)
                avg_auc = np.nanmean(fold_aucs)
                avg_ic = np.nanmean(fold_ics)
                avg_bss_val = np.nanmean(fold_bss)
                std_score = np.nanstd(fold_scores)
                avg_logloss = np.nanmean(fold_logloss)
                avg_accuracy = np.nanmean(fold_accuracy)

                results[name] = avg_score
                detailed_metrics[name] = {
                    "score": avg_score,
                    "AUC": avg_auc,
                    "IC": avg_ic,
                    "BSS": avg_bss_val,
                    "std_score": std_score,
                    "LogLoss": avg_logloss,
                    "Accuracy": avg_accuracy
                }
                tprint(f"  {name}: Score={avg_score:.4f}  AUC={avg_auc:.4f}  IC={avg_ic:.4f}  BSS={avg_bss_val:.4f}  LogLoss={avg_logloss:.4f}  Acc={avg_accuracy:.4f}  Std={std_score:.4f}")

            except Exception as e:
                tprint(f"  {name} Failed: {e}")
                results[name] = -float("inf")

        self.detailed_metrics = detailed_metrics

        if not results:
            raise ValueError("All models failed in race")

        best_name = max(results, key=results.get)
        self.best_model_name = best_name
        self.metrics = results
        if best_name in detailed_metrics:
            dm = detailed_metrics[best_name]
            tprint(f"Race Winner: {best_name} (Score={dm['score']:.4f}, AUC={dm['AUC']:.4f}, IC={dm['IC']:.4f}, BSS={dm['BSS']:.4f})")
        else:
            tprint(f"Race Winner: {best_name} (Score={results[best_name]:.4f})")

        # 2. Generate OOF predictions with best model (for meta model)
        tprint(f"Generating OOF predictions with {best_name}...")
        oof_probs = np.full(len(y), np.nan, dtype=np.float32)
        oof_candidates = self._get_candidates(race_mode=True)
        oof_model = oof_candidates[best_name]
        for train_idx, val_idx in cached_splits:
            if use_numpy:
                X_tr, X_val = X_np[train_idx], X_np[val_idx]
            else:
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = safe_slice(y, train_idx)
            w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
            oof_model.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_probs[val_idx] = oof_model.predict_proba(X_val)[:, 1]
        # Fill any remaining NaN with 0.5 (neutral)
        oof_probs = np.nan_to_num(oof_probs, nan=0.5)
        self.oof_probs = oof_probs
        tprint(f"OOF predictions: mean={np.mean(oof_probs):.4f}, std={np.std(oof_probs):.4f}")

        # Calculate Winner OOF Metrics
        try:
            oof_auc = roc_auc_score(y, oof_probs)
            oof_logloss = log_loss(y, oof_probs)
            oof_accuracy = accuracy_score(y, oof_probs > 0.5)
            if returns is not None and np.std(oof_probs) > 1e-9 and np.std(returns) > 1e-9:
                oof_ic = np.corrcoef(rankdata(oof_probs), rankdata(returns))[0, 1]
            else:
                oof_ic = 0.0
            tprint(f"Winner OOF Metrics: AUC={oof_auc:.4f}  IC={oof_ic:.4f}  LogLoss={oof_logloss:.4f}  Acc={oof_accuracy:.4f}")
        except Exception as e:
            tprint(f"Error calculating OOF metrics: {e}")

        # Recap
        tprint("\n=== Model Race Recap ===")
        tprint(f"{'Model':<15} {'Score':>8} {'AUC':>8} {'IC':>8} {'BSS':>8} {'LogLoss':>8} {'Acc':>8} {'Std':>8}")
        tprint("-" * 85)

        sorted_models = sorted(detailed_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
        for name, m in sorted_models:
             tprint(f"{name:<15} {m['score']:8.4f} {m['AUC']:8.4f} {m['IC']:8.4f} {m['BSS']:8.4f} {m['LogLoss']:8.4f} {m['Accuracy']:8.4f} {m['std_score']:8.4f}")
        tprint("========================\n")

        # 3. Final Retraining & Calibration
        tprint(f"Retraining {best_name} on full data (full config)...")
        final_candidates = self._get_candidates(race_mode=False)
        final_base = final_candidates[best_name]

        self.best_model = CalibratedClassifierCV(
            estimator=final_base,
            method='isotonic',
            cv=3
        )
        self.best_model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        tprint(f"Entering function: predict_proba in model_race.py")
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        return self.best_model.predict_proba(X)

    def predict(self, X):
        # Return probability class 1
        tprint(f"Entering function: predict in model_race.py")
        return self.predict_proba(X)[:, 1]
