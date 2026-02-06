import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from scipy.stats import rankdata, spearmanr
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from extreme_price_movements.utils import tprint
from extreme_price_movements.purged_cv import PurgedKFold

class Float64Wrapper(BaseEstimator, ClassifierMixin):
    """Wraps a classifier so predict_proba / decision_function always return float64.
    Some estimators (e.g. XGBoost) return float32 predictions by default."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        if sample_weight is not None:
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.estimator.predict(X)

    def decision_function(self, X):
        if hasattr(self.estimator, 'decision_function'):
            return np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        return self.predict_proba(X)[:, 1]

    def get_params(self, deep=True):
        return {"estimator": self.estimator}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self


class ScaledLogisticRegression(LogisticRegression):
    """
    Wrapper to apply StandardScaler internally, ensuring sample_weight 
    is correctly passed to fit (bypassing Pipeline limitations with CalibratedClassifierCV).
    """
    def __init__(self, class_weight=None, **kwargs):
        super().__init__(class_weight=class_weight, **kwargs)
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



def _softsign(x):
    return x / (1.0 + np.abs(x))


def _zscore(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)


def _max_drawdown(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    if ec.size < 2:
        return 0.0
    peak = np.maximum.accumulate(ec)
    dd = peak - ec
    mdd = np.nanmax(dd)
    return float(mdd) if np.isfinite(mdd) else 0.0


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _normalize_bss(bss):
    bss_capped = float(np.clip(bss, -1.0, 1.0))
    return _clip01((bss_capped + 1.0) / 2.0)


def _position_from_prob(
    p,
    *,
    mode="linear",            # "linear" | "centered" | "rank" | "sigmoid"
    min_pos=0.0,
    max_pos=1.0,
    center=0.5,               # used by "centered"
    sigmoid_k=6.0,            # used by "sigmoid"
):
    """
    Map model probability/score to position size in [min_pos, max_pos] (or symmetric if min_pos<0).
    Assumes p is higher => stronger signal.
    """
    p = np.asarray(p, dtype=float)

    if mode == "linear":
        s = p
    elif mode == "centered":
        # -1..1 if p in [0,1], centered at `center`
        s = (p - center) / (0.5 if center == 0.5 else max(1e-12, max(center, 1 - center)))
        s = np.clip(s, -1.0, 1.0)
        # map -1..1 to min_pos..max_pos if symmetric bounds; else just scale into range
        if min_pos < 0.0 and max_pos > 0.0:
            # symmetric-ish: allow both long/short
            # s=-1 -> min_pos, s=+1 -> max_pos
            s = (s + 1.0) / 2.0
        else:
            # long-only
            s = (s + 1.0) / 2.0
    elif mode == "rank":
        # robust to calibration; uses cross-sectional rank
        r = np.argsort(np.argsort(p)).astype(float)
        s = r / max(1.0, (len(p) - 1.0))
    elif mode == "sigmoid":
        # squashes extremes, useful if p isn't calibrated
        s = 1.0 / (1.0 + np.exp(-sigmoid_k * (p - 0.5)))
    else:
        raise ValueError(f"Unknown position mode: {mode}")

    # Final scale to [min_pos, max_pos]
    s = np.clip(s, 0.0, 1.0) if (min_pos >= 0.0 and max_pos >= 0.0) else np.clip(s, 0.0, 1.0)
    pos = min_pos + (max_pos - min_pos) * s
    return pos


def calculate_selection_score(
    y_true,
    y_prob,
    trade_returns,
    *,
    # ---- Position sizing ----
    size_mode="rank",       # "linear"|"rank"|"sigmoid"|"centered"
    min_pos=0.0,              # long-only default. set -1..1 for long/short with centered mode
    max_pos=1.0,
    size_center=0.5,          # centered mode
    sigmoid_k=6.0,            # sigmoid mode
    size_clip=(0.0, 1.0),     # additional clip safety
    leverage=3.0,             # scalar multiplier on position size

    # ---- Realized metric ----
    cost_per_trade=0.005,       # cost per unit position (so pos*cost). Default set to 0.5%
    use_log_equity=False,
    annualization_factor=None,
    dd_penalty=0.25,
    coverage_penalty=0.10,

    # ---- Utility-weighted IC ----
    utility_clip=3.0,
    utility_power=1.0,
    ic_cap=0.10,

    # ---- BSS ----
    bss_min_prev=0.02,
    bss_cap_ref_min=1e-6,

    # ---- Composite weights ----
    w_realized=0.55,
    w_uic=0.35,
    w_bss=0.10,
):
    """
    Same as v2, but realized returns are *position-sized*:
        sized_return_i = position_i * trade_return_i - abs(position_i)*cost_per_trade

    Position is derived from y_prob (or its rank), enabling bet sizing / leverage effects to
    influence the realized metric and the utility-weighted IC.
    """
    y_true = np.asarray(y_true) if y_true is not None else None
    y_prob = np.asarray(y_prob, dtype=float)
    r = np.asarray(trade_returns, dtype=float)

    n = min(len(y_prob), len(r), len(y_true) if y_true is not None else len(y_prob))
    y_prob = y_prob[:n]
    r = r[:n]
    if y_true is not None:
        y_true = y_true[:n]

    m = np.isfinite(y_prob) & np.isfinite(r)
    if y_true is not None:
        m = m & np.isfinite(y_true)

    y_prob_m = y_prob[m]
    r_m = r[m]
    y_true_m = y_true[m].astype(int) if y_true is not None else None

    metrics = {"N": int(n), "N_valid": int(m.sum())}
    if metrics["N_valid"] < 5:
        metrics.update({
            "Position_Mean": 0.0,
            "Position_AbsMean": 0.0,
            "Realized_Metric": 0.0,
            "Realized_Score": 0.0,
            "Utility_IC": 0.0,
            "Utility_IC_Score": 0.0,
            "BSS": 0.0,
            "BSS_Score": 0.5,
            "Selection_Score": 0.0,
        })
        return metrics

    # -------------------------
    # 0) Position sizing
    # -------------------------
    pos = _position_from_prob(
        y_prob_m,
        mode=size_mode,
        min_pos=min_pos,
        max_pos=max_pos,
        center=size_center,
        sigmoid_k=sigmoid_k,
    )
    pos = np.asarray(pos, dtype=float) * float(leverage)
    pos = np.clip(pos, float(size_clip[0]), float(size_clip[1])) if size_clip is not None else pos

    # -------------------------
    # 1) Position-sized realized returns
    # -------------------------
    # -------------------------
    # 1) Position-sized realized returns
    # -------------------------
    # Cost scales with absolute exposure
    sized_r = pos * r_m - np.abs(pos) * float(cost_per_trade)

    # Equity curve
    if use_log_equity:
        equity = np.nancumsum(sized_r)
        peak = np.maximum.accumulate(equity)
        # DD as % from peak (assuming returns are log-returns approx)
        # 1 - exp(equity - peak)
        dd_series = 1.0 - np.exp(equity - peak)
    else:
        equity = np.nancumprod(1.0 + sized_r)
        peak = np.maximum.accumulate(equity)
        dd_series = (peak - equity) / np.maximum(peak, 1e-12)

    mu = float(np.nanmean(sized_r))
    sd = float(np.nanstd(sized_r, ddof=1)) if len(sized_r) > 1 else 0.0
    sharpe_per_trade = mu / (sd + 1e-12)
    sharpe = sharpe_per_trade * np.sqrt(float(annualization_factor)) if annualization_factor is not None else sharpe_per_trade

    mdd_pct = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
    # MDD Penalty: directly penalize % DD. 
    # e.g. 20% DD => 0.2 penalty. 
    # We want robust score in [0,1].
    
    # Coverage: Fraction of ACTIVE trades (abs(pos) > epsilon)
    active_mask = np.abs(pos) > 1e-6
    coverage = np.mean(active_mask)

    # Score components
    # Map Sharpe to [0,1] robustly (softsign centered at 0?)
    # softsign(x) = x / (1+|x|). Maps -inf->-1, inf->1.
    # We want 0->0.5? Or just positive sharpe focus?
    # Let's use user's softsign logic but clearer:
    # realized_raw in [-1, 1]
    
    # cov_term: penalize low coverage.
    cov_term = np.clip(coverage, 0.0, 1.0)
    
    # Penalty logic: 
    # Score = (Softsign(Sharpe) - P_dd * MDD + P_cov * Coverage) normalized?
    # Let's keep it simple:
    # Base = 0.5 + 0.5 * softsign(Sharpe)  (in 0..1)
    # Penalties subtraction
    
    base_realized = 0.5 * (1.0 + (sharpe / (1.0 + abs(sharpe))))
    
    # Penalize MDD: limit impact to say 0.3
    # If MDD=0.2 (20%), penalty = 0.2 * dd_penalty
    dd_impact = dd_penalty * mdd_pct
    
    # Reward coverage: small bump if coverage is high, or penalty if low?
    # User had coverage_penalty * cov_term (additive).
    # Let's say we want at least 5% coverage. 
    # If coverage < 0.05 => penalty.
    # Simpler: just add weighted coverage.
    cov_impact = coverage_penalty * cov_term

    realized_score = np.clip(base_realized - dd_impact + cov_impact, 0.0, 1.0)

    metrics["Position_Mean"] = float(np.nanmean(pos))
    metrics["Position_AbsMean"] = float(np.nanmean(np.abs(pos)))
    metrics["Sized_Return_Mean"] = mu
    metrics["Sized_Return_Std"] = sd
    metrics["Sharpe"] = float(sharpe)
    metrics["Max_Drawdown"] = mdd_pct
    metrics["Coverage"] = float(coverage)
    metrics["Realized_Score"] = float(realized_score)

    # -------------------------
    # 2) Utility-weighted IC (prob vs utility of *UNIT* returns)
    # -------------------------
    # Avoid feedback loop: Usage of Sized returns inflates IC.
    # Use r_m (raw unit trade returns) for utility calculation.
    ur = _zscore(r_m)
    ur = np.clip(ur, -float(utility_clip), float(utility_clip))
    # Utility function: focuses on tails of the *market opportunity*
    u = np.sign(ur) * (np.abs(ur) ** float(utility_power))

    if np.nanstd(y_prob_m) < 1e-12 or np.nanstd(u) < 1e-12:
        uic = 0.0
    else:
        uic = spearmanr(y_prob_m, u, nan_policy="omit").correlation
        uic = 0.0 if (uic is None or not np.isfinite(uic)) else float(uic)

    # Sigmoid scaling for IC to prevent hard cap saturation
    # sigmoid: 2 / (1 + exp(-x/s)) - 1
    # scale s ~ 0.05 so that IC=0.10 => score ~ 0.76, IC=0.2 => score ~ 0.96
    s_ic = 0.08
    uic_score = 2.0 / (1.0 + np.exp(-max(0.0, uic) / s_ic)) - 1.0
    uic_score = np.clip(uic_score, 0.0, 1.0)

    metrics["Utility_IC"] = float(uic)
    metrics["Utility_IC_Score"] = float(uic_score)

    # -------------------------
    # 3) Brier Skill Score (calibration)
    # -------------------------
    bss = 0.0
    bss_score = 0.5
    if y_true_m is not None:
        p = np.clip(y_prob_m, 0.0, 1.0)
        prev = float(np.mean(y_true_m)) if len(y_true_m) else 0.0
        if bss_min_prev < prev < (1.0 - bss_min_prev):
            try:
                bs = float(brier_score_loss(y_true_m, p))
                bs_ref = float(brier_score_loss(y_true_m, np.full_like(p, prev)))
                bs_ref = max(bs_ref, float(bss_cap_ref_min))
                bss = 1.0 - (bs / bs_ref)
                if not np.isfinite(bss):
                    bss = 0.0
                bss_score = _normalize_bss(bss)
            except Exception:
                bss, bss_score = 0.0, 0.5

    metrics["BSS"] = float(bss)
    metrics["BSS_Score"] = float(bss_score)

    # -------------------------
    # 4) Composite
    # -------------------------
    # Adjust weights to de-emphasize BSS if using Rank sizing
    # E.g. 0.60, 0.35, 0.05
    sel = (
        float(w_realized) * metrics["Realized_Score"] +
        float(w_uic) * metrics["Utility_IC_Score"] +
        float(w_bss) * metrics["BSS_Score"]
    )
    metrics["Selection_Score"] = float(np.clip(sel, 0.0, 1.0))

    # Diagnostic keys
    if y_true_m is not None:
        try:
             if len(np.unique(y_true_m)) > 1:
                 metrics["AUC"] = float(roc_auc_score(y_true_m, y_prob_m))
             else:
                 metrics["AUC"] = 0.5
        except:
             metrics["AUC"] = 0.5
    else:
        metrics["AUC"] = 0.5
        
    metrics["IC"] = metrics["Utility_IC"]
    # Or keep original IC? The new logic uses Utility IC.
    # We can add standard IC too.
    try:
        std_ic = spearmanr(y_prob_m, r_m, nan_policy="omit").correlation
        metrics["Standard_IC"] = float(std_ic) if (std_ic is not None and np.isfinite(std_ic)) else 0.0
    except:
        metrics["Standard_IC"] = 0.0
    
    return metrics



class ModelRace(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="long", n_splits=5, race_sample_frac=0.5, race_early_stopping_rounds=50):
        self.kind = kind
        self.n_splits = n_splits
        self.race_sample_frac = race_sample_frac
        self.race_early_stopping_rounds = race_early_stopping_rounds
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.detailed_metrics = {}
        self.oof_probs = None

    def _compute_pos_weight(self, y):
        # Inverse class frequency
        return (len(y) - y.sum()) / max(1, y.sum())

    def _subsample_indices(self, indices, frac, seed=42):
        if frac >= 1.0:
            return indices
        np.random.seed(seed)
        n_samples = int(len(indices) * frac)
        return np.random.choice(indices, n_samples, replace=False)

    def _get_candidates(self, race_mode=True):
        # Configure models
        # ExtraTrees, XGBoost, LightGBM, CatBoost
        
        candidates = {}
        
        # 1. ExtraTrees
        et_params = {
            "n_estimators": 200 if race_mode else 800,
            "max_depth": 7,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42
        }
        candidates["extratrees"] = Float64Wrapper(ExtraTreesClassifier(**et_params))

        # 2. XGBoost
        xgb_params = {
            "n_estimators": 200 if race_mode else 800,
            "max_depth": 4,
            "learning_rate": 0.05,
            "reg_lambda": 5.0,              # L2 (default=1 is often too weak)
            "reg_alpha": 0.0,               # keep 0 initially
            "min_child_weight": 20,
            "tree_method": "hist",
            "gamma": 1.0,                   
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "min_samples_split": 100,
            "enable_categorical": False
        }
        candidates["xgboost"] = Float64Wrapper(XGBClassifier(**xgb_params))

        # 3. LightGBM
        lgb_params = {
            "n_estimators": 200 if race_mode else 800,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "lambda_l1": 0.0,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1
        }
        candidates["lightgbm"] = Float64Wrapper(LGBMClassifier(**lgb_params))

        # 4. CatBoost
        cb_params = {
            "iterations": 200 if race_mode else 800,
            "l2_leaf_reg": 10.0,        
            "random_strength": 1.0,     
            "bagging_temperature": 1.0,         
            "depth": 4,
            "learning_rate": 0.05,
            "verbose": 0,
            "thread_count": -1,
            "random_seed": 42,
            "allow_writing_files": False
        }
        candidates["catboost"] = Float64Wrapper(CatBoostClassifier(**cb_params))
        
        return candidates

    def _fit_model(self, model, X_tr, y_tr, X_val=None, y_val=None, sample_weight=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        pos_weight = self._compute_pos_weight(y_tr)

        if isinstance(model, ScaledLogisticRegression):
            # Safe to set because we updated __init__
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
                    # early_stopping_rounds deprecated in fit, use constructor or callbacks if needed
                    # For simple race, we can omit it or relying on constructor
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
        # Cast y and sample_weight to float64 for consistent dtype handling
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if returns is None:
            returns = y
        else:
            returns = np.asarray(returns, dtype=np.float64)

        # Optimize: Convert to numpy once if possible (and suitable for all models)
        # ExtraTrees/XGBoost prefer numpy. CatBoost handles both but numpy is fine if no categorical features.
        # We assume numeric features here.
        X_np = X
        use_numpy = False
        if hasattr(X, "iloc"):
            try:
                # Float32 for memory; Float64Wrapper ensures predict_proba returns float64
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
                    # No subsampling — use identical splits for race and OOF
                    if use_numpy:
                        X_tr, X_val = X_np[train_idx], X_np[val_idx]
                    else:
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

                    y_tr = safe_slice(y, train_idx)
                    y_val = safe_slice(y, val_idx)
                    w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
                    ret_val = safe_slice(returns, val_idx)

                    # Raw model fit — no CalibratedClassifierCV.
                    # Treat predict_proba output as rank scores, not calibrated probabilities.
                    estimator = clone(model)
                    if w_tr is not None:
                        estimator.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        estimator.fit(X_tr, y_tr)
                    
                    probs = estimator.predict_proba(X_val)[:, 1]

                    # w_bss=0: raw outputs are rank scores, BSS is meaningless
                    metrics = calculate_selection_score(y_val, probs, ret_val, w_bss=0.0, w_realized=0.60, w_uic=0.40)
                    fold_scores.append(metrics["Selection_Score"])
                    fold_aucs.append(metrics["AUC"])
                    fold_ics.append(metrics["IC"])
                    fold_bss.append(metrics["BSS"])

                    try:
                        fold_logloss.append(log_loss(y_val, np.clip(probs, 1e-7, 1-1e-7)))
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
            
            # Raw model fit — same as race, no calibration wrapper
            estimator = clone(oof_model)
            if w_tr is not None:
                estimator.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                estimator.fit(X_tr, y_tr)
            oof_probs[val_idx] = estimator.predict_proba(X_val)[:, 1]
        # Fill any remaining NaN with 0.5 (neutral)
        oof_probs = np.nan_to_num(oof_probs, nan=0.5)
        self.oof_probs = oof_probs
        tprint(f"OOF predictions: mean={np.mean(oof_probs):.4f}, std={np.std(oof_probs):.4f}")

        # Effective sample size diagnostic
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            n_eff = (np.sum(sw) ** 2) / np.sum(sw ** 2)
            tprint(f"Weight diagnostics: n={len(sw)}, n_eff={n_eff:.0f} ({100*n_eff/len(sw):.0f}%), mean={np.mean(sw):.3f}, std={np.std(sw):.3f}, p95={np.percentile(sw,95):.3f}")

        # Calculate Winner OOF Metrics (rank-based, not calibration-dependent)
        try:
            oof_auc = roc_auc_score(y, oof_probs)
            oof_logloss = log_loss(y, np.clip(oof_probs, 1e-7, 1-1e-7))
            oof_accuracy = accuracy_score(y, oof_probs > 0.5)
            if returns is not None and np.std(oof_probs) > 1e-9 and np.std(returns) > 1e-9:
                oof_ic = np.corrcoef(rankdata(oof_probs), rankdata(returns))[0, 1]
            else:
                oof_ic = 0.0
            # OOF selection score (same weights as race)
            oof_sel = calculate_selection_score(y, oof_probs, returns, w_bss=0.0, w_realized=0.60, w_uic=0.40)
            tprint(f"Winner OOF Metrics: AUC={oof_auc:.4f}  IC={oof_ic:.4f}  LogLoss={oof_logloss:.4f}  Acc={oof_accuracy:.4f}  SelScore={oof_sel['Selection_Score']:.4f}")
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

        # 3. Final Retraining (raw model, no calibration wrapper)
        # Calibration is harmful with small n and uncalibrated objectives.
        # Output is treated as rank score by downstream (engine, backtest).
        tprint(f"Retraining {best_name} on full data (full config)...")
        final_candidates = self._get_candidates(race_mode=False)
        final_base = final_candidates[best_name]
        self.best_model = clone(final_base)
        if sample_weight is not None:
            self.best_model.fit(X, y, sample_weight=sample_weight)
        else:
            self.best_model.fit(X, y)

        return self

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        return self.best_model.predict_proba(X)

    def predict(self, X):
        # Return probability class 1 (rank score, not calibrated probability)
        return self.predict_proba(X)[:, 1]
