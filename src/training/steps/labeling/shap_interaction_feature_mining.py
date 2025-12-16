import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_warning

try:
    import shap

    _SHAP_AVAILABLE = True
except Exception:
    shap = None
    _SHAP_AVAILABLE = False

try:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    _ML_AVAILABLE = True
except Exception:
    lgb = None
    RandomForestClassifier = None
    roc_auc_score = None
    accuracy_score = None
    TimeSeriesSplit = None
    _ML_AVAILABLE = False


def _sanitize(name: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _make_name(prefix: str, transform: str, a: str, b: str) -> str:
    na = _sanitize(a)
    nb = _sanitize(b)
    base = f"{prefix}{transform}__{na}__{nb}"
    if len(base) <= 220:
        return base
    try:
        import hashlib

        h = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}{transform}__{na[:40]}__{nb[:40]}__h{h}"
    except Exception:
        return base[:220]


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    denom = np.where(np.abs(b) > eps, b, np.where(b >= 0, eps, -eps))
    return a / denom


def apply_interaction_definitions(
    X: pd.DataFrame,
    interaction_defs: List[Dict[str, Any]],
    fillna_value: float = 0.0,
    eps: float = 1e-8,
) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not interaction_defs:
        return pd.DataFrame(index=X.index)

    out = pd.DataFrame(index=X.index)
    for d in interaction_defs:
        try:
            name = str(d.get("name") or "")
            f1 = str(d.get("feature_a") or "")
            f2 = str(d.get("feature_b") or "")
            tr = str(d.get("transform") or "prod")
            if not name or f1 not in X.columns or f2 not in X.columns:
                continue
            if name in X.columns or name in out.columns:
                continue
            a = pd.to_numeric(X[f1], errors="coerce").fillna(fillna_value).to_numpy(dtype=float)
            b = pd.to_numeric(X[f2], errors="coerce").fillna(fillna_value).to_numpy(dtype=float)
            if tr == "prod":
                v = a * b
            elif tr == "diff":
                v = a - b
            elif tr == "ratio":
                v = _safe_div(a, b, eps=eps)
            else:
                continue
            out[name] = np.asarray(v, dtype=np.float32)
        except Exception:
            continue
    return out


def _stratified_sample_index(y: pd.Series, n: int, random_state: int) -> pd.Index:
    rs = np.random.RandomState(random_state)
    if n <= 0 or len(y) == 0:
        return y.index[:0]
    try:
        y_non = y.dropna()
        classes = list(pd.unique(y_non))
        if len(classes) < 2:
            return y_non.sample(n=min(n, len(y_non)), random_state=random_state).index
        per = max(1, int(n // len(classes)))
        sampled: List[Any] = []
        for c in classes:
            idx = y_non[y_non == c].index
            if len(idx) == 0:
                continue
            take = min(per, len(idx))
            sampled.extend(list(rs.choice(idx, size=take, replace=False)))
        rem = int(n - len(sampled))
        if rem > 0:
            pool = [i for i in y_non.index if i not in set(sampled)]
            if pool:
                sampled.extend(list(rs.choice(pool, size=min(rem, len(pool)), replace=False)))
        return pd.Index(sampled[:n])
    except Exception:
        return y.dropna().sample(n=min(n, int(y.notna().sum())), random_state=random_state).index


def _create_model(model_type: str, random_state: int, cfg: Dict[str, Any], key: str) -> Any:
    n_estimators = int(cfg.get(key, cfg.get("n_estimators", 300)))
    if str(model_type).lower() == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=int(cfg.get("rf_max_depth", 10)),
            random_state=random_state,
            n_jobs=-1,
        )
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=float(cfg.get("lgbm_learning_rate", 0.05)),
        max_depth=int(cfg.get("lgbm_max_depth", 6)),
        num_leaves=int(cfg.get("lgbm_num_leaves", 31)),
        subsample=float(cfg.get("lgbm_subsample", 0.8)),
        colsample_bytree=float(cfg.get("lgbm_colsample_bytree", 0.8)),
        reg_alpha=float(cfg.get("lgbm_reg_alpha", 0.1)),
        reg_lambda=float(cfg.get("lgbm_reg_lambda", 0.1)),
        random_state=random_state,
        verbosity=-1,
    )


def _coerce_shap_values(values: Any, n_features: int) -> Optional[np.ndarray]:
    try:
        if isinstance(values, list) and values:
            arr = np.asarray(values[-1])
        else:
            arr = np.asarray(getattr(values, "values", values))
        if arr.ndim == 3:
            arr = np.abs(arr).mean(axis=2)
        if arr.ndim != 2:
            return None
        if arr.shape[1] == n_features + 1:
            arr = arr[:, :-1]
        return arr
    except Exception:
        return None


def _coerce_interactions(values: Any, n_features: int) -> Optional[np.ndarray]:
    try:
        if isinstance(values, list) and values:
            arr = np.asarray(values[-1])
        else:
            arr = np.asarray(values)
        if arr.ndim == 4:
            arr = np.abs(arr).mean(axis=3)
        if arr.ndim != 3:
            return None
        if arr.shape[1] == n_features + 1 and arr.shape[2] == n_features + 1:
            arr = arr[:, :-1, :-1]
        return arr
    except Exception:
        return None


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, w: Optional[np.ndarray] = None) -> Optional[float]:
    try:
        mask = ~(np.isnan(y_true) | np.isnan(y_score))
        if int(mask.sum()) < 20:
            return None
        yt = y_true[mask]
        ps = y_score[mask]
        if np.unique(yt).size < 2:
            return None
        if w is None:
            return float(roc_auc_score(yt, ps))
        return float(roc_auc_score(yt, ps, sample_weight=np.asarray(w)[mask]))
    except Exception:
        return None


def evaluate_interaction_uplift(
    X_base: pd.DataFrame,
    y: pd.Series,
    w: Optional[pd.Series],
    interaction_defs: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    random_state: int,
    embargo_pct: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"enabled": True}
    if not interaction_defs:
        out["enabled"] = False
        out["reason"] = "no_interactions"
        return out

    n_splits = int(cfg.get("eval_folds", 3))
    gap = int(max(0, int(len(X_base) * float(embargo_pct))))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    X0 = X_base.fillna(float(cfg.get("fillna_value", 0.0)))
    X_int = apply_interaction_definitions(X0, interaction_defs, fillna_value=float(cfg.get("fillna_value", 0.0)))
    X1 = pd.concat([X0, X_int], axis=1)

    auc0: List[float] = []
    auc1: List[float] = []
    for tr, te in tscv.split(X0):
        Xtr0, Xte0 = X0.iloc[tr], X0.iloc[te]
        Xtr1, Xte1 = X1.iloc[tr], X1.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        wtr = None
        wte = None
        if w is not None:
            wtr = np.asarray(w.iloc[tr], dtype=float)
            wte = np.asarray(w.iloc[te], dtype=float)

        m0 = _create_model(cfg.get("model_type", "lgbm"), random_state, cfg, "eval_n_estimators")
        kw = {}
        if wtr is not None:
            kw["sample_weight"] = wtr
        m0.fit(Xtr0, ytr, **kw)
        p0 = m0.predict_proba(Xte0)[:, 1] if hasattr(m0, "predict_proba") else m0.predict(Xte0)

        m1 = _create_model(cfg.get("model_type", "lgbm"), random_state, cfg, "eval_n_estimators")
        kw1 = {}
        if wtr is not None:
            kw1["sample_weight"] = wtr
        m1.fit(Xtr1, ytr, **kw1)
        p1 = m1.predict_proba(Xte1)[:, 1] if hasattr(m1, "predict_proba") else m1.predict(Xte1)

        if y.nunique() <= 2 and roc_auc_score is not None:
            a0 = _safe_auc(yte.to_numpy(dtype=float), np.asarray(p0, dtype=float), wte)
            a1 = _safe_auc(yte.to_numpy(dtype=float), np.asarray(p1, dtype=float), wte)
        else:
            a0 = float(accuracy_score(yte, (np.asarray(p0) > 0.5).astype(int)))
            a1 = float(accuracy_score(yte, (np.asarray(p1) > 0.5).astype(int)))
        if a0 is not None:
            auc0.append(float(a0))
        if a1 is not None:
            auc1.append(float(a1))

    out["baseline_mean"] = float(np.mean(auc0)) if auc0 else None
    out["augmented_mean"] = float(np.mean(auc1)) if auc1 else None
    out["delta_mean"] = (out["augmented_mean"] - out["baseline_mean"]) if (out["augmented_mean"] is not None and out["baseline_mean"] is not None) else None
    out["n_folds"] = int(min(len(auc0), len(auc1)))
    return out


def mine_shap_interaction_feature_defs(
    X: pd.DataFrame,
    y: pd.Series,
    target_sample_weight: Optional[pd.Series] = None,
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    embargo_pct: float = 0.01,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = dict(config or {})
    info: Dict[str, Any] = {"enabled": bool(_ML_AVAILABLE and _SHAP_AVAILABLE)}

    if not info["enabled"]:
        info["reason"] = "missing_dependencies"
        return [], info

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    y = y.reindex(X.index)

    w = None
    if target_sample_weight is not None:
        if not isinstance(target_sample_weight, pd.Series):
            w = pd.Series(target_sample_weight, index=X.index)
        else:
            w = target_sample_weight.reindex(X.index)
        w = w.fillna(1.0)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if w is not None:
        w = w.loc[valid]

    Xn = X.select_dtypes(include=[np.number])
    cols = [c for c in Xn.columns if not str(c).lower().startswith("signal_")]
    Xn = Xn[cols]

    if Xn.shape[1] < 2 or len(Xn) < int(cfg.get("min_samples", 200)):
        info["enabled"] = False
        info["reason"] = "insufficient_data"
        info["n_samples"] = int(len(Xn))
        info["n_features"] = int(Xn.shape[1])
        return [], info

    prefix = str(cfg.get("prefix", "shap_int__"))
    fillna_value = float(cfg.get("fillna_value", 0.0))
    sample_size = int(min(int(cfg.get("sample_size", 800)), len(Xn)))

    regime_cfg = cfg.get("regime") if isinstance(cfg.get("regime"), dict) else {}
    require_regime_pair = bool(regime_cfg.get("require_regime_feature", False))
    exclude_regime_regime = bool(regime_cfg.get("exclude_regime_regime", True))
    regime_prefixes = regime_cfg.get("prefixes")
    if not isinstance(regime_prefixes, list) or not regime_prefixes:
        regime_prefixes = ["regime_leaf_"]
    regime_prefixes = [str(p) for p in regime_prefixes if str(p)]

    def _is_regime_feature(name: str) -> bool:
        try:
            s = str(name)
            return any(s.startswith(p) for p in regime_prefixes)
        except Exception:
            return False

    idx = _stratified_sample_index(y, sample_size, random_state)
    Xs = Xn.loc[idx].fillna(fillna_value)
    ys = y.loc[idx]
    ws = w.loc[idx] if w is not None else None

    if verbose:
        tprint_info(f"   [SHAP interactions] mining on samples={int(len(Xs))}, features={int(Xs.shape[1])}")

    model_type = str(cfg.get("model_type", "lgbm"))
    kw = {}
    if ws is not None:
        kw["sample_weight"] = np.asarray(ws, dtype=float)

    m0 = _create_model(model_type, random_state, cfg, "main_n_estimators")
    m0.fit(Xs, ys, **kw)
    expl0 = shap.TreeExplainer(m0)
    shap_vals = expl0.shap_values(Xs)
    arr0 = _coerce_shap_values(shap_vals, Xs.shape[1])
    if arr0 is None:
        info["enabled"] = False
        info["reason"] = "shap_values_failed"
        return [], info

    main_imp = np.mean(np.abs(arr0), axis=0)
    top_k = int(min(int(cfg.get("top_main_features", 25)), Xs.shape[1]))
    top_features = (
        pd.Series(main_imp, index=Xs.columns).sort_values(ascending=False).head(top_k).index.tolist()
    )

    Xs_top = Xs[top_features]
    m1 = _create_model(model_type, random_state, cfg, "interaction_n_estimators")
    m1.fit(Xs_top, ys, **kw)
    expl1 = shap.TreeExplainer(m1)
    inter_vals = expl1.shap_interaction_values(Xs_top)
    arr1 = _coerce_interactions(inter_vals, Xs_top.shape[1])
    if arr1 is None:
        info["enabled"] = False
        info["reason"] = "shap_interaction_failed"
        return [], info

    pairs: List[Tuple[float, str, str]] = []
    nfeat = int(Xs_top.shape[1])
    for i in range(nfeat):
        for j in range(i + 1, nfeat):
            try:
                a_name = str(top_features[i])
                b_name = str(top_features[j])
                if require_regime_pair:
                    if not (_is_regime_feature(a_name) or _is_regime_feature(b_name)):
                        continue
                if exclude_regime_regime and _is_regime_feature(a_name) and _is_regime_feature(b_name):
                    continue
                s = float(np.mean(np.abs(arr1[:, i, j])))
                if not np.isfinite(s):
                    continue
                pairs.append((s, a_name, b_name))
            except Exception:
                continue

    pairs.sort(key=lambda x: x[0], reverse=True)
    max_pairs = int(cfg.get("max_pairs", 30))
    transforms = cfg.get("transforms", ["prod"])
    if not isinstance(transforms, list):
        transforms = [str(transforms)]

    max_new = int(cfg.get("max_new_features", 20))
    defs: List[Dict[str, Any]] = []
    for score, a, b in pairs[:max_pairs]:
        for tr in transforms:
            if len(defs) >= max_new:
                break
            name = _make_name(prefix, str(tr), a, b)
            defs.append(
                {
                    "name": name,
                    "transform": str(tr),
                    "feature_a": a,
                    "feature_b": b,
                    "shap_interaction_score": float(score),
                }
            )
        if len(defs) >= max_new:
            break

    info["n_candidates"] = int(len(pairs))
    info["top_features"] = list(top_features)
    info["interaction_defs"] = list(defs)

    try:
        eval_cfg = dict(cfg)
        info["evaluation"] = evaluate_interaction_uplift(
            X_base=Xn,
            y=y,
            w=w,
            interaction_defs=defs,
            cfg=eval_cfg,
            random_state=random_state,
            embargo_pct=embargo_pct,
        )
    except Exception:
        info["evaluation"] = {"enabled": False, "reason": "evaluation_failed"}

    min_delta = cfg.get("min_delta_auc")
    try:
        if min_delta is not None:
            min_delta = float(min_delta)
            delta = info.get("evaluation", {}).get("delta_mean")
            if delta is not None and float(delta) < min_delta:
                if verbose:
                    tprint_warning(
                        f"   [SHAP interactions] delta_auc={float(delta):.4f} < min_delta_auc={min_delta:.4f}; skipping"
                    )
                return [], {**info, "skipped": True}
    except Exception:
        pass

    return defs, info
