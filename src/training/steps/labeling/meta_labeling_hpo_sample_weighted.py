"""Meta-Labeling HPO Sample Weighted Step.

This step orchestrates the Layer 2 -> Layer 3 pipeline for label generation
and calibration.

Layer 2: Regime-Conditional Geometry Optimization (LabelBasedLayer2)
- Optimizes Barrier Geometries (TP/SL/Horizon) per barrier family.
- Selects diverse geometries.
- Generates Bagged OOF Labels and Weights (K-Fold OOF for analytics).
- Also generates Production Geometries (Full Fit).

Layer 3: Calibration & Meta-Model (LabelBasedLayer3)
- Feature Engineering on Layer 2 outputs (Disagreement, Volatility).
- Weights adjustment using Magnitude and Layer 1 weights.
- Calibrated Probability generation using LGBM + Isotonic Regression (K-Fold OOF).
- Final Model training on full dataset.

Layer 4: Position Sizing & Diagnostics (LabelBasedLayer4)
- Converts calibrated probabilities to position sizes.
- Computes advanced diagnostics (Edge Monotonicity, Bet Efficiency).
- Generates final sized events for backtesting.

This replaces the legacy HierarchicalParameterOptimizer loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_success, tprint_warning, tprint_info, tprint_error

from src.training.steps.labeling.label_based_layer_0 import run_layer0_kalman_vwap
from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm, plot_diagnostics
# Import Layer 4 (Risk Filter) and Layer 5 (Position Sizing)
from src.training.steps.labeling.label_based_layer_4 import (
    Layer4RiskFilter,
    compute_layer4_regime_features,
    train_layer4_oof,
)
from src.training.steps.labeling.oos_integration import (
    should_run_oos,
    create_oos_split,
    generate_oos_predictions,
    run_oos_layer5_evaluation,
    compare_oos_vs_oof,
    save_oos_results,
    validate_oos_config,
)
from src.training.steps.labeling.label_based_layer_5 import Layer5PositionSizer
from src.training.steps.labeling.label_based_layer_1 import run_layer1_optimization
from src.training.steps.labeling.generate_weights_per_label import (
    compute_uniqueness,
    compute_horizon_consistency,
    generate_weights_per_label,
)

from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
)

from src.utils.ml_common.transaction_costs import get_transaction_cost


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _select_econ_deadband_auto(
    returns: pd.Series,
    coverage_min: float,
    coverage_max: float,
    grid_size: int,
    min_class_count: int,
) -> tuple[float, dict]:
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return 0.0, {"reason": "no_returns"}

    abs_r = r.abs().to_numpy(dtype=float, copy=False)
    abs_r = abs_r[np.isfinite(abs_r)]
    if abs_r.size == 0:
        return 0.0, {"reason": "no_finite_returns"}

    try:
        cov_min = float(coverage_min)
    except Exception:
        cov_min = 0.10
    try:
        cov_max = float(coverage_max)
    except Exception:
        cov_max = 0.30
    if (not np.isfinite(cov_min)) or cov_min < 0.0:
        cov_min = 0.10
    if (not np.isfinite(cov_max)) or cov_max <= cov_min:
        cov_max = float(min(0.95, cov_min + 0.20))

    try:
        n_grid = int(grid_size)
    except Exception:
        n_grid = 41
    n_grid = int(max(11, min(n_grid, 201)))

    try:
        min_cnt = int(min_class_count)
    except Exception:
        min_cnt = 200
    min_cnt = int(max(20, min_cnt))

    target_cov = float(0.5 * (cov_min + cov_max))

    qs = np.linspace(0.0, 0.99, n_grid)
    best = {
        "score": -np.inf,
        "deadband": 0.0,
        "coverage": float("nan"),
        "pos_rate": float("nan"),
        "n_pos": 0,
        "n_neg": 0,
        "cohens_d": float("nan"),
        "snr": float("nan"),
        "mean_pos": float("nan"),
        "mean_neg": float("nan"),
    }

    def _entropy_norm(p: float) -> float:
        try:
            p = float(p)
        except Exception:
            return 0.0
        if not np.isfinite(p):
            return 0.0
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        return float(np.clip(h / np.log(2.0), 0.0, 1.0))

    r_np = r.to_numpy(dtype=float, copy=False)
    best_in_band = None
    for q in qs:
        try:
            d = float(np.quantile(abs_r, float(q)))
        except Exception:
            continue
        if (not np.isfinite(d)) or d < 0.0:
            continue

        pos = r_np > d
        neg = r_np < -d
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(neg))
        n_lab = int(n_pos + n_neg)
        cov = float(n_lab / max(1, int(r_np.size)))
        if n_pos < min_cnt or n_neg < min_cnt:
            continue

        rp = r_np[pos]
        rn = r_np[neg]
        rp = rp[np.isfinite(rp)]
        rn = rn[np.isfinite(rn)]
        if rp.size < min_cnt or rn.size < min_cnt:
            continue

        mean_pos = float(np.mean(rp))
        mean_neg = float(np.mean(rn))
        std_pos = float(np.std(rp))
        std_neg = float(np.std(rn))
        pooled = float(np.sqrt(0.5 * (std_pos * std_pos + std_neg * std_neg)) + 1e-12)
        cohens_d = float((mean_pos - mean_neg) / pooled)
        snr = float(mean_pos / (std_pos + 1e-12))

        pos_rate = float(n_pos / max(1, n_lab))
        ent = _entropy_norm(pos_rate)

        base_score = float(max(0.0, mean_pos) * max(0.0, cohens_d) * max(0.0, snr) * (0.25 + 0.75 * ent))
        if not np.isfinite(base_score):
            continue

        if cov < cov_min or cov > cov_max:
            base_score = float(base_score - 10.0 * abs(float(cov) - float(target_cov)))

        if base_score > best["score"]:
            best.update(
                {
                    "score": float(base_score),
                    "deadband": float(d),
                    "coverage": float(cov),
                    "pos_rate": float(pos_rate),
                    "n_pos": int(n_pos),
                    "n_neg": int(n_neg),
                    "cohens_d": float(cohens_d),
                    "snr": float(snr),
                    "mean_pos": float(mean_pos),
                    "mean_neg": float(mean_neg),
                }
            )

        if cov_min <= cov <= cov_max:
            if best_in_band is None or base_score > float(best_in_band.get("score", -np.inf)):
                best_in_band = dict(best)

    if best_in_band is not None and np.isfinite(float(best_in_band.get("score", -np.inf))):
        return float(best_in_band["deadband"]), best_in_band

    if not np.isfinite(float(best.get("score", -np.inf))):
        return 0.0, {"reason": "no_candidate"}

    return float(best["deadband"]), best


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        n_in_bin = int(np.sum(in_bin))
        if n_in_bin == 0:
            continue
        p_mean = float(np.mean(y_prob[in_bin]))
        o_mean = float(np.mean(y_true[in_bin]))
        ece += (n_in_bin / y_true.size) * abs(p_mean - o_mean)
    return float(ece)


def _compute_layer2_metrics(
    l2_labels: pd.Series,
    l2_returns: pd.Series,
    l2_weights: pd.Series,
) -> Dict[str, Any]:
    labels = l2_labels.copy()
    returns = l2_returns.copy()
    weights = l2_weights.copy()

    labeled_mask = labels.notna()
    n_total = int(labels.shape[0])
    n_labeled = int(labeled_mask.sum())

    if n_labeled > 0:
        pos_rate = float((labels[labeled_mask] == 1).mean())
    else:
        pos_rate = float("nan")

    ret_eval = returns[labeled_mask].astype(float)
    w_eval = weights[labeled_mask].astype(float)

    def _entropy_norm(w: np.ndarray) -> float:
        w = np.asarray(w)
        w = w[np.isfinite(w)]
        if w.size == 0:
            return float("nan")
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s <= 0:
            return float("nan")
        p = w / s
        ent = -float(np.sum(p * np.log(p + 1e-12)))
        return float(ent / np.log(max(2, p.size)))

    metrics: Dict[str, Any] = {
        "n_total": n_total,
        "n_labeled": n_labeled,
        "coverage": float(n_labeled / n_total) if n_total > 0 else float("nan"),
        "pos_rate": pos_rate,
        "mean_return": _safe_float(ret_eval.mean()),
        "median_return": _safe_float(ret_eval.median()),
        "std_return": _safe_float(ret_eval.std(ddof=0)),
        "mean_weight": _safe_float(w_eval.mean()),
        "median_weight": _safe_float(w_eval.median()),
        "std_weight": _safe_float(w_eval.std(ddof=0)),
        "weight_entropy_norm": _entropy_norm(w_eval.values),
    }
    return metrics


def _compute_layer3_metrics(oof_export: pd.DataFrame, target_col: str, prob_col: str = "meta_prob") -> Dict[str, Any]:
    if target_col not in oof_export.columns or prob_col not in oof_export.columns:
        return {"error": f"missing_columns target_col={target_col} prob_col={prob_col}"}

    y = pd.to_numeric(oof_export[target_col], errors="coerce").astype(float)
    p = pd.to_numeric(oof_export[prob_col], errors="coerce").astype(float)
    mask = y.notna() & p.notna()
    if int(mask.sum()) == 0:
        return {"error": "no_valid_oof_rows"}

    y_eval_raw = np.asarray(y[mask].values, dtype=float)
    p_eval = np.asarray(p[mask].values, dtype=float)
    p_eval = np.clip(p_eval, 1e-6, 1.0 - 1e-6)

    # Binarize target if it's not already strictly {0,1}.
    # Layer3 internally thresholds continuous targets at 0.5 for classification metrics.
    try:
        uniq = np.unique(y_eval_raw[np.isfinite(y_eval_raw)])
        is_binary_like = bool(
            (uniq.size > 0)
            and (uniq.size <= 2)
            and bool(np.all(np.isin(uniq, [0.0, 1.0])))
        )
    except Exception:
        is_binary_like = False

    if is_binary_like:
        y_eval = y_eval_raw
    else:
        y_eval = (y_eval_raw > 0.5).astype(float)

    auc = float("nan")
    try:
        if int(np.unique(y_eval).size) >= 2:
            auc = float(roc_auc_score(y_eval.astype(int), p_eval))
    except Exception:
        pass

    ll = float("nan")
    try:
        ll = float(log_loss(y_eval.astype(int), p_eval, labels=[0, 1]))
    except Exception:
        pass

    brier = float("nan")
    try:
        brier = float(brier_score_loss(y_eval.astype(int), p_eval))
    except Exception:
        pass

    ece = _compute_ece(y_eval.astype(float), p_eval, n_bins=10)

    metrics: Dict[str, Any] = {
        "n_eval": int(mask.sum()),
        "auc": auc,
        "log_loss": ll,
        "brier": brier,
        "ece": float(ece),
        "target_is_binary_like": bool(is_binary_like),
        "prob_mean": _safe_float(np.mean(p_eval)),
        "prob_std": _safe_float(np.std(p_eval)),
    }
    return metrics


def _compute_layer4_sweep_metrics(
    oof_df: pd.DataFrame,
    target_col: str,
    p_col: str,
    return_col: str,
    p_min: float,
    p_max: float,
    gamma: float,
    transaction_cost: float,
) -> Dict[str, Any]:
    try:
        p_vec = pd.to_numeric(oof_df[p_col], errors="coerce").to_numpy(dtype=float, copy=False)
        p_gate = np.isfinite(p_vec) & (p_vec >= float(p_min))
        n_prob_ge_pmin = int(np.sum(p_gate))

        raw_rets = pd.to_numeric(oof_df[return_col], errors="coerce").to_numpy(dtype=float, copy=False)
        try:
            tx_cost = float(transaction_cost)
        except Exception:
            tx_cost = 0.0
        if (not np.isfinite(tx_cost)) or abs(float(tx_cost)) <= 1e-15:
            net_rets = raw_rets
        else:
            net_rets = raw_rets - float(tx_cost)
        gate_rets = net_rets[p_gate]
        gate_rets = gate_rets[np.isfinite(gate_rets)]
        gate_avg_pnl = float(np.mean(gate_rets)) if gate_rets.size > 0 else float("nan")
        gate_total_pnl = float(np.nansum(gate_rets)) if gate_rets.size > 0 else 0.0
        gate_wins = gate_rets[gate_rets > 0.0]
        gate_losses = gate_rets[gate_rets < 0.0]
        gate_gross_profit = float(np.sum(gate_wins)) if gate_wins.size > 0 else 0.0
        gate_gross_loss = float(-np.sum(gate_losses)) if gate_losses.size > 0 else 0.0
        gate_pf = float(gate_gross_profit / (gate_gross_loss + 1e-12)) if gate_gross_loss > 0.0 else float("nan")
        gate_win_rate = float(gate_wins.size / gate_rets.size) if gate_rets.size > 0 else float("nan")

        gate_auc = float("nan")
        gate_ece = float("nan")
        gate_n_eval = int(gate_rets.size)
        try:
            if int(n_prob_ge_pmin) > 0:
                l3_gate = _compute_layer3_metrics(oof_df.loc[p_gate], target_col=target_col, prob_col=p_col)
                if isinstance(l3_gate, dict):
                    gate_auc = float(l3_gate.get("auc")) if l3_gate.get("auc") is not None else float("nan")
                    gate_ece = float(l3_gate.get("ece")) if l3_gate.get("ece") is not None else float("nan")
                    gate_n_eval = int(l3_gate.get("n_eval")) if l3_gate.get("n_eval") is not None else gate_n_eval
        except Exception:
            pass

        sizer = Layer5PositionSizer(
            oof_df=oof_df,
            p_col=p_col,
            target_col=target_col,
            return_col=return_col,
            p_min=float(p_min),
            p_max=float(p_max),
            gamma=float(gamma),
            transaction_cost=float(tx_cost),
        )
        sizes = sizer.calculate_sizing().to_numpy(dtype=float, copy=False)
        pnl = sizes * net_rets
        trade_mask = np.asarray(sizes > 1e-4, dtype=bool)
        traded_pnl = pnl[trade_mask]
        traded_pnl = traded_pnl[np.isfinite(traded_pnl)]

        n_trades = int(np.sum(trade_mask))
        avg_trade_pnl = float(np.mean(traded_pnl)) if traded_pnl.size > 0 else float("nan")

        wins = traded_pnl[traded_pnl > 0.0]
        losses = traded_pnl[traded_pnl < 0.0]
        gross_profit = float(np.sum(wins)) if wins.size > 0 else 0.0
        gross_loss = float(-np.sum(losses)) if losses.size > 0 else 0.0
        profit_factor = float(gross_profit / (gross_loss + 1e-12)) if gross_loss > 0.0 else float("nan")
        win_rate = float(wins.size / traded_pnl.size) if traded_pnl.size > 0 else float("nan")

        return {
            "p_min": float(p_min),
            "n_prob_ge_pmin": int(n_prob_ge_pmin),
            "gate_n_eval": int(gate_n_eval),
            "gate_auc": float(gate_auc) if np.isfinite(gate_auc) else float("nan"),
            "gate_ece": float(gate_ece) if np.isfinite(gate_ece) else float("nan"),
            "gate_avg_pnl": float(gate_avg_pnl) if np.isfinite(gate_avg_pnl) else float("nan"),
            "gate_total_pnl": float(gate_total_pnl) if np.isfinite(gate_total_pnl) else float("nan"),
            "gate_profit_factor": float(gate_pf) if np.isfinite(gate_pf) else float("nan"),
            "gate_win_rate": float(gate_win_rate) if np.isfinite(gate_win_rate) else float("nan"),
            "n_trades": int(n_trades),
            "avg_trade_pnl": float(avg_trade_pnl) if np.isfinite(avg_trade_pnl) else float("nan"),
            "total_pnl": float(np.nansum(pnl)),
            "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float("nan"),
            "win_rate": float(win_rate) if np.isfinite(win_rate) else float("nan"),
        }
    except Exception:
        return {
            "p_min": float(p_min),
            "n_prob_ge_pmin": 0,
            "gate_n_eval": 0,
            "gate_auc": float("nan"),
            "gate_ece": float("nan"),
            "gate_avg_pnl": float("nan"),
            "gate_total_pnl": float("nan"),
            "gate_profit_factor": float("nan"),
            "gate_win_rate": float("nan"),
            "n_trades": 0,
            "avg_trade_pnl": float("nan"),
            "total_pnl": float("nan"),
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
        }


def _write_layer5_pmin_sweep(
    outcomes_dir: Path,
    l4_input: pd.DataFrame,
    target_col: str,
    p_col: str,
    return_col: str,
    p_max: float,
    gamma: float,
    transaction_cost: float,
    p_min_values: Optional[List[float]] = None,
) -> Optional[str]:
    try:
        if p_min_values is None:
            p_min_values = [float(x) for x in np.round(np.linspace(0.2, 0.6, 9), 3)]
        rows: List[Dict[str, Any]] = []
        for p_min in list(p_min_values or []):
            rows.append(
                _compute_layer4_sweep_metrics(
                    oof_df=l4_input,
                    target_col=target_col,
                    p_col=p_col,
                    return_col=return_col,
                    p_min=float(p_min),
                    p_max=float(p_max),
                    gamma=float(gamma),
                    transaction_cost=float(transaction_cost),
                )
            )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = outcomes_dir / f"layer5_pmin_sweep_{ts}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return str(out_path)
    except Exception:
        return None


def _write_unified_label_based_report(
    outcomes_dir: Path,
    context: Dict[str, Any],
    layer2_metrics: Dict[str, Any],
    layer3_metrics: Dict[str, Any],
    layer4_risk_metrics: Dict[str, Any],
    layer5_metrics: Dict[str, Any],
) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outcomes_dir / f"label_based_unified_report_{ts}.csv"
    json_path = outcomes_dir / f"label_based_unified_report_{ts}.json"
    md_path = outcomes_dir / f"label_based_unified_report_{ts}.md"

    rows: List[Dict[str, Any]] = []
    for layer_name, metrics in [
        ("label_based_layer_2", layer2_metrics),
        ("label_based_layer_3", layer3_metrics),
        ("label_based_layer_4", layer4_risk_metrics),
        ("label_based_layer_5", layer5_metrics),
    ]:
        for k, v in (metrics or {}).items():
            rows.append({"layer": layer_name, "metric": str(k), "value": v})

    report_df = pd.DataFrame(rows)
    report_df.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "context": context,
        "layer2": layer2_metrics,
        "layer3": layer3_metrics,
        "layer4": layer4_risk_metrics,
        "layer5": layer5_metrics,
        "csv_path": str(csv_path),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    try:
        def _fmt_kv_block(title: str, metrics: Dict[str, Any]) -> List[str]:
            lines = [f"## {title}\n"]
            if not isinstance(metrics, dict) or not metrics:
                lines.append("- (empty)\n")
                return lines
            for k in sorted(metrics.keys(), key=lambda x: str(x)):
                try:
                    v = metrics.get(k)
                except Exception:
                    continue
                try:
                    if isinstance(v, float):
                        if np.isfinite(v):
                            lines.append(f"- {k}: {v:.6g}\n")
                        else:
                            lines.append(f"- {k}: {v}\n")
                    else:
                        lines.append(f"- {k}: {v}\n")
                except Exception:
                    lines.append(f"- {k}: {str(v)}\n")
            return lines

        md_lines: List[str] = [
            "# Label-Based Unified Report\n",
            f"- generated_at: {payload.get('generated_at')}\n",
            f"- csv_path: {str(csv_path)}\n",
            f"- json_path: {str(json_path)}\n",
            "\n## Context\n",
        ]
        if isinstance(context, dict):
            for k in sorted(context.keys(), key=lambda x: str(x)):
                try:
                    md_lines.append(f"- {k}: {context.get(k)}\n")
                except Exception:
                    continue
        else:
            md_lines.append(f"- context: {context}\n")

        md_lines.append("\n")
        md_lines.extend(_fmt_kv_block("Layer2 Metrics", layer2_metrics))
        md_lines.append("\n")
        md_lines.extend(_fmt_kv_block("Layer3 Metrics", layer3_metrics))
        md_lines.append("\n")
        md_lines.extend(_fmt_kv_block("Layer4 Metrics", layer4_risk_metrics))
        md_lines.append("\n")
        md_lines.extend(_fmt_kv_block("Layer5 Metrics", layer5_metrics))

        md_path.write_text("".join(md_lines))
    except Exception:
        pass

    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}


def _normalize_labeling_start_at(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    aliases = {
        "0": "layer0",
        "1": "layer1",
        "2": "layer2",
        "3": "layer3",
        "4": "layer4",
        "layer0": "layer0",
        "stage0": "layer0",
        "kalman": "layer0",
        "weighting": "layer2",
        "trading": "layer2",
        "model": "layer3",
    }
    return aliases.get(s, s)


def _read_layer3_oof_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def _timeframe_to_minutes(timeframe: Any) -> int:
    try:
        s = str(timeframe).strip().lower()
    except Exception:
        return 15

    if s.endswith("m"):
        try:
            return int(float(s[:-1]))
        except Exception:
            return 15
    if s.endswith("h"):
        try:
            return int(float(s[:-1]) * 60)
        except Exception:
            return 60
    if s.endswith("d"):
        try:
            return int(float(s[:-1]) * 1440)
        except Exception:
            return 1440
    try:
        return int(float(s))
    except Exception:
        return 15


class MetaLabelingHPOSampleWeightedStep(BaseStep):
    """
    Orchestrates the Layer 2 -> Layer 3 -> Layer 4 meta-labeling pipeline.
    """

    def _load_market_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load market data using BaseStep's standard multi-stage loading strategy."""
        market_data, _source = self.load_market_data_or_fail(
            config,
            pipeline_state={},
            allow_config_override=True,
            light_mode_filter=True,
        )
        return market_data

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            config: Configuration dictionary.
        """
        outcomes_dir = Path(config.get("outcomes_dir", "outcomes"))
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not isinstance(config, dict):
                config = {}
        except Exception:
            config = {}
        try:
            if config.get("run_timestamp") is None:
                config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        except Exception:
            config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        start_at_raw = config.get("labeling_hpo_start_at")
        start_at = _normalize_labeling_start_at(start_at_raw)
        explicit_start = start_at is not None

        # Load market data (using standard BaseStep mechanism)
        market_data = self._load_market_data(config)
        
        if market_data is None or market_data.empty:
            tprint_error("Failed to load market data.")
            return {"success": False}

        # ---------------------------------------------------------------------
        # PRE-PROCESSING: Volatility and Regime Generation
        # (Must happen before splitting into train_data / oos_data)
        # ---------------------------------------------------------------------
        if ("volatility_1d" not in market_data.columns) or bool(
            pd.to_numeric(market_data.get("volatility_1d"), errors="coerce").isna().all()
        ):
            close_series = pd.to_numeric(market_data.get("close"), errors="coerce")
            returns_1 = close_series.pct_change().replace([np.inf, -np.inf], np.nan)

            tf_minutes = _timeframe_to_minutes(config.get("timeframe", "15m"))
            bars_per_day = int(max(2, round(1440 / max(1, int(tf_minutes)))))
            window = int(max(2, config.get("volatility_1d_window", bars_per_day)))

            vol_1d = returns_1.rolling(window).std()
            vol_1d = pd.to_numeric(vol_1d, errors="coerce")
            vol_1d = vol_1d.replace([np.inf, -np.inf], np.nan)
            vol_1d = vol_1d.ffill().bfill()
            try:
                fallback = float(np.nanmedian(np.abs(returns_1.values)))
            except Exception:
                fallback = 0.0
            if (not np.isfinite(fallback)) or fallback <= 0.0:
                fallback = 0.0
            vol_1d = vol_1d.fillna(fallback)
            market_data["volatility_1d"] = vol_1d.astype(float)

        vol_regime_existing = None
        if "vol_regime" in market_data.columns:
            vol_regime_existing = market_data["vol_regime"]

        if (vol_regime_existing is None) or bool(
            pd.Series(vol_regime_existing, index=market_data.index)
            .astype(str)
            .replace({"nan": np.nan})
            .isna()
            .all()
        ):
            vol_1d_series = pd.to_numeric(market_data.get("volatility_1d"), errors="coerce")
            # CAUSAL REGIME CALCULATION (Fixing Global Quantile Leakage)
            # User requested "rolling growing window" -> Expanding Window
            tf_minutes = _timeframe_to_minutes(config.get("timeframe", "15m"))
            bars_per_day = int(max(2, round(1440 / max(1, int(tf_minutes)))))
            
            # Volatility Regime
            try:
                # Expanding window: Anchored at start, grows indefinitely (min_periods=1 day)
                vol_thr = vol_1d_series.expanding(min_periods=bars_per_day).quantile(
                    float(config.get("vol_regime_high_q", 0.67))
                )
                # Fill initial NaNs with the first valid threshold (backward fill)
                # Only leaked for the first 'bars_per_day' samples (1 day), acceptable startup cost
                vol_thr = vol_thr.bfill()
            except Exception:
                vol_thr = float(vol_1d_series.median())

            if isinstance(vol_thr, pd.Series):
                market_data["vol_regime"] = np.where(vol_1d_series >= vol_thr, "High", "Low")
            else:
                market_data["vol_regime"] = np.where(vol_1d_series >= vol_thr, "High", "Low")

        trend_regime_existing = None
        if "trend_regime" in market_data.columns:
            trend_regime_existing = market_data["trend_regime"]

        if (trend_regime_existing is None) or bool(
            pd.Series(trend_regime_existing, index=market_data.index)
            .astype(str)
            .replace({"nan": np.nan})
            .isna()
            .all()
        ):
            close_series = pd.to_numeric(market_data.get("close"), errors="coerce")
            returns_1 = close_series.pct_change().replace([np.inf, -np.inf], np.nan)
            vol_1d_series = pd.to_numeric(market_data.get("volatility_1d"), errors="coerce").replace(
                0.0, np.nan
            )

            tf_minutes = _timeframe_to_minutes(config.get("timeframe", "15m"))
            bars_per_day = int(max(2, round(1440 / max(1, int(tf_minutes)))))
            trend_window = int(max(2, config.get("trend_regime_window", bars_per_day)))
            trend_mu = returns_1.rolling(trend_window).mean()
            trend_score = (trend_mu.abs() / (vol_1d_series.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
            
            # Trend Regime (Causal - Expanding)
            try:
                trend_thr = trend_score.expanding(min_periods=bars_per_day).quantile(
                    float(config.get("trend_regime_high_q", 0.67))
                )
                trend_thr = trend_thr.bfill()
            except Exception:
                trend_thr = float(np.nanmedian(trend_score.values))

            if isinstance(trend_thr, pd.Series):
                market_data["trend_regime"] = np.where(trend_score >= trend_thr, "High", "Low")
            else:
                market_data["trend_regime"] = np.where(trend_score >= trend_thr, "High", "Low")

        market_data["vol_regime"] = market_data["vol_regime"].astype(str).fillna("Low")
        market_data["trend_regime"] = market_data["trend_regime"].astype(str).fillna("Low")
        
        # ---------------------------------------------------------------------
        # OOS TEST: Check if we should run OOS evaluation
        # ---------------------------------------------------------------------
        oos_results = None
        train_data = market_data
        oos_data = None
        
        if should_run_oos(config, market_data):
            tprint_info(">>> OOS Test Enabled - Creating temporal split...")
            try:
                oos_config = validate_oos_config(config)
                train_data, oos_data, split_date = create_oos_split(
                    market_data,
                    oos_days=oos_config['oos_days'],
                    min_train_days=oos_config['min_train_days']
                )
                tprint_success(f"OOS split created: train={len(train_data)} bars, test={len(oos_data)} bars")
            except Exception as e:
                tprint_warning(f"Failed to create OOS split: {e}")
                oos_data = None

        # ---------------------------------------------------------------------
        # LAYER 0: Kalman/RTS + VWAP optimization
        # ---------------------------------------------------------------------
        layer0_bundle_path = outcomes_dir / "layer0_kalman_bundle.joblib"
        layer0_params: Dict[str, Any] = {}
        layer0_artifacts: Dict[str, Any] = {}

        if (not explicit_start) or (start_at == "layer0"):
            run_opt = bool((not layer0_bundle_path.exists()) or (start_at == "layer0"))
            market_data, layer0_payload = run_layer0_kalman_vwap(
                symbol=str(config.get("symbol", "")),
                timeframe=str(config.get("timeframe", "")),
                market_data=market_data,
                config=config,
                outcomes_dir=outcomes_dir,
                bundle_path=layer0_bundle_path,
                run_optimization=run_opt,
                train_data=train_data,  # Use training data only
            )
            layer0_params = dict(layer0_payload.get("best_params", {}) or {})
            layer0_artifacts["layer0_kalman_bundle"] = str(layer0_bundle_path)
            layer0_artifacts["layer0_params"] = dict(layer0_params)

        else:
            if layer0_bundle_path.exists():
                try:
                    l0_bundle = joblib.load(layer0_bundle_path)
                    layer0_params = dict(l0_bundle.get("best_params", {}) or {})
                except Exception:
                    layer0_params = {}


        layer1_bundle_path = outcomes_dir / "layer1_weighting_bundle.joblib"
        target_sample_weight = config.get("target_sample_weight")

        if (start_at != "layer1") and (target_sample_weight is None) and layer1_bundle_path.exists():
            try:
                l1_bundle = joblib.load(layer1_bundle_path)
                target_sample_weight = l1_bundle.get("target_sample_weight")
                config["target_sample_weight"] = target_sample_weight
            except Exception:
                pass

        if ((not explicit_start) or (start_at == "layer1")) and (target_sample_weight is None or not layer1_bundle_path.exists()):
            close_series = pd.to_numeric(market_data.get("close"), errors="coerce")
            returns_1 = close_series.pct_change()
            if "volatility_1d" in market_data.columns:
                vol_1d = pd.to_numeric(market_data["volatility_1d"], errors="coerce")
            else:
                vol_1d = returns_1.rolling(96).std()

            snr = returns_1.abs() / vol_1d.replace(0, np.nan)
            event_idx = market_data.index[snr > 0.5]

            signals = pd.DataFrame(index=market_data.index)
            signals["consensus"] = 0.0
            try:
                dirs = np.sign(returns_1.reindex(event_idx)).replace([np.inf, -np.inf], 0.0)
                dirs = dirs.fillna(0.0)
                dirs = dirs.replace(0.0, 1.0)
                signals.loc[event_idx, "consensus"] = dirs.astype(float)
            except Exception:
                signals.loc[event_idx, "consensus"] = 1.0

            vol_aligned = vol_1d.reindex(market_data.index).fillna(0.0)
            profit_thr = (vol_aligned * 2.0).astype(float).clip(lower=0.008)
            stop_thr = (vol_aligned * 1.0).astype(float).clip(lower=0.004)

            baseline_returns, _, _, _, _, _, _, _ = compute_realized_returns(
                df=market_data,
                signals=signals,
                profit_threshold=profit_thr,
                stop_threshold=stop_thr,
                horizon=12,
                transaction_cost=float(config.get("transaction_cost", 0.001)),
                min_event_spacing=0,
                volatility_series=vol_aligned,
            )

            baseline_evt = baseline_returns.reindex(event_idx).astype(float)
            baseline_evt = baseline_evt.replace([np.inf, -np.inf], np.nan).dropna()

            best_weighting_params = run_layer1_optimization(
                symbol=str(config.get("symbol", "")),
                timeframe=str(config.get("timeframe", "")),
                market_data=market_data,
                labels=baseline_evt,
                n_trials=int(config.get("layer1_n_trials", 60)),
                objective_mode=str(config.get("layer1_objective_mode", "proxy")),
            )

            tf_minutes = _timeframe_to_minutes(config.get("timeframe", "15m"))
            t_events = pd.DatetimeIndex(baseline_evt.index)
            t1 = pd.Series(
                data=t_events + pd.Timedelta(minutes=int(tf_minutes) * 12),
                index=t_events,
            )
            uniq = compute_uniqueness(t1, events_index=t1.index, market_index=market_data.index)
            uniq_arr = uniq.values if isinstance(uniq, pd.Series) else np.asarray(uniq, dtype=float)

            cons = compute_horizon_consistency(close_series.astype(float), horizon=12)
            cons_arr = cons.reindex(t_events).fillna(0.0).values

            vol_proxy = vol_aligned.reindex(t_events).fillna(0.0).values

            best_weighting_params_local = (
                dict(best_weighting_params)
                if isinstance(best_weighting_params, dict)
                else {}
            )
            if "transaction_cost" not in best_weighting_params_local:
                try:
                    best_weighting_params_local["transaction_cost"] = float(get_transaction_cost(config))
                except Exception:
                    best_weighting_params_local["transaction_cost"] = float(get_transaction_cost(None))
            w_evt = generate_weights_per_label(
                returns=baseline_evt.values,
                t_events=t_events,
                consistency_scores=cons_arr,
                uniqueness_scores=uniq_arr,
                vol_proxy=vol_proxy,
                **best_weighting_params_local,
            )

            target_sample_weight_series = pd.Series(1.0, index=market_data.index, dtype=float)
            target_sample_weight_series.loc[t_events] = pd.Series(w_evt, index=t_events).reindex(t_events).values
            target_sample_weight_series = target_sample_weight_series.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            config["target_sample_weight"] = target_sample_weight_series
            target_sample_weight = target_sample_weight_series

            joblib.dump(
                {
                    "best_params": best_weighting_params,
                    "target_sample_weight": target_sample_weight_series,
                    "n_events": int(len(t_events)),
                },
                layer1_bundle_path,
            )


        layer2_bundle_path = outcomes_dir / "layer2_oof_bundle.joblib"
        l2_labels = None
        l2_score = None
        l2_confidence = None
        l2_returns = None
        l2_weights = None
        individual_geos = None
        events_df = None
        selected_trials = None

        # Check for Layer 2 substep flag
        layer2_substep = config.get("layer2_substep")
        if layer2_substep:
            start_at = "layer2" # Force context if running substep

        if start_at in ("layer3", "layer4") and not bool(config.get('force_hpo', False)) and not layer2_substep:
            if not layer2_bundle_path.exists():
                raise FileNotFoundError(
                    f"Missing Layer2 bundle at {layer2_bundle_path}. Run with --labeling-hpo-start-at layer2 first."
                )
            l2_bundle = joblib.load(layer2_bundle_path)
            l2_labels = l2_bundle["l2_labels"]
            l2_score = l2_bundle.get("l2_score")
            l2_confidence = l2_bundle.get("l2_confidence")
            l2_returns = l2_bundle["l2_returns"]
            l2_weights = l2_bundle["l2_weights"]
            individual_geos = l2_bundle["individual_geos"]
            individual_variances = l2_bundle.get("individual_variances")
            events_df = l2_bundle["events_df"]
            selected_trials = l2_bundle.get("selected_trials")
            l2_quality_weights = l2_bundle.get("l2_quality_weights")
            if not isinstance(selected_trials, list):
                selected_trials = []
            if len(selected_trials) == 0:
                tprint_warning("Layer2 production geometries are empty (loaded from bundle).")

            try:
                if isinstance(l2_labels, pd.Series):
                    y = pd.to_numeric(l2_labels, errors="coerce").astype(float)
                    uniq = np.unique(y.dropna().values)
                    is_binary = bool(
                        (uniq.size <= 2)
                        and bool(np.all(np.isin(uniq, [0.0, 1.0])))
                    )
                    if not is_binary:
                        l2_labels = (y >= 0.5).astype(float).where(y.notna())
            except Exception:
                pass
        else:
            # ---------------------------------------------------------
            # LAYER 2: Geometry Optimization & Bagged Labeling
            # ---------------------------------------------------------
            tprint_info(">>> Executing Layer 2: Geometry Optimization (OOF & Full)...")

            layer2 = LabelBasedLayer2(
                transaction_cost=get_transaction_cost(config),
                n_trials=int(config.get('layer2_n_trials', 30)),
                n_splits=int(config.get('layer2_n_splits', 3)),
                verbose=True,
                force_hpo=bool(config.get('force_hpo', False))
            )

            # Granular Execution Paths
            if layer2_substep == "prepare":
                tprint_info(">>> Layer 2 Substep: PREPARE")
                df_prep, events_df, X_events, global_probe_features = layer2.prepare_data_and_events(train_data)
                joblib.dump(
                    {
                        "df": df_prep,
                        "events_df": events_df,
                        "X_events": X_events,
                        "global_probe_features": global_probe_features
                    },
                    outcomes_dir / "layer2_prep_bundle.joblib"
                )
                tprint_success("Layer 2 Prep Bundle Saved.")
                return {"success": True}

            elif layer2_substep == "optimize":
                tprint_info(">>> Layer 2 Substep: OPTIMIZE")
                try:
                    prep = joblib.load(outcomes_dir / "layer2_prep_bundle.joblib")
                except Exception:
                    raise FileNotFoundError("Missing layer2_prep_bundle.joblib. Run --layer2-substep prepare first.")

                production_geometries, production_selected_features = layer2.optimize_production_geometries(
                    prep["df"], prep["events_df"], global_probe_features=prep.get("global_probe_features")
                )

                joblib.dump(
                    {
                        "production_geometries": production_geometries,
                        "production_selected_features": production_selected_features
                    },
                    outcomes_dir / "layer2_optim_bundle.joblib"
                )
                tprint_success("Layer 2 Optim Bundle Saved.")
                return {"success": True}

            elif layer2_substep == "oof":
                tprint_info(">>> Layer 2 Substep: OOF")
                try:
                    prep = joblib.load(outcomes_dir / "layer2_prep_bundle.joblib")
                    optim = joblib.load(outcomes_dir / "layer2_optim_bundle.joblib")
                    # Handle legacy optim bundle if it was just list
                    if isinstance(optim, list):
                         production_geometries = optim
                         production_selected_features = []
                    else:
                         production_geometries = optim.get("production_geometries")
                         production_selected_features = optim.get("production_selected_features")
                except Exception:
                    raise FileNotFoundError("Missing prep/optim bundles. Run prepare and optimize first.")

                # Run OOF
                oof_results = layer2.run_oof_analytics(
                    prep["df"], prep["events_df"], production_geometries,
                    global_probe_features=prep.get("global_probe_features"),
                    production_selected_features=production_selected_features
                )

                # Merge into full output format
                l2_output = {
                    **oof_results,
                    "events_df": prep["events_df"],
                    "selected_trials": [asdict(t) for t in production_geometries],
                    "production_selected_features": production_selected_features or [],
                }
                # Fall through to standard saving logic (below)

            elif layer2_substep == "report":
                tprint_info(">>> Layer 2 Substep: REPORT")
                # Need everything loaded
                try:
                    prep = joblib.load(outcomes_dir / "layer2_prep_bundle.joblib")
                    optim = joblib.load(outcomes_dir / "layer2_optim_bundle.joblib")
                    l2_output = joblib.load(layer2_bundle_path) # Need OOF results

                    if isinstance(optim, list):
                        production_geometries = optim
                    else:
                        production_geometries = optim.get("production_geometries")
                except Exception:
                    raise FileNotFoundError("Missing artifacts for report generation.")

                # Reconstruct oof_results dict from flat output for generate_reports signature
                oof_res_reconstructed = {
                    'l2_score': l2_output.get('l2_score'),
                    'l2_label': l2_output.get('l2_labels'), # Fixed key from l2_label to l2_labels
                    'weights': l2_output.get('l2_weights'),
                    'individual_geometries': l2_output.get('individual_geos'),
                    'oof_returns': l2_output.get('l2_returns'),
                    'tree_stats': l2_output.get('tree_stats'), # Pass tree stats
                }
                layer2.generate_reports(prep["df"], prep["events_df"], production_geometries, oof_res_reconstructed)
                tprint_success("Layer 2 Reports Regenerated.")
                return {"success": True}

            else:
                # Standard Monolithic Run
                l2_output = layer2.run(train_data)

            # Unpack Layer 2 Artifacts (OOF for Training/Analytics)
            l2_score = l2_output.get('l2_score')
            l2_confidence = l2_output.get('l2_confidence')
            l2_labels = l2_output.get('l2_label', l2_output.get('oof_labels'))
            l2_returns = l2_output['oof_returns']
            l2_weights = l2_output['weights']
            l2_quality_weights = l2_output.get('quality_weights')
            individual_geos = l2_output['individual_geometries']
            individual_variances = l2_output.get('individual_variances') # Unpack Variances
            events_df = l2_output['events_df']
            selected_trials = l2_output.get('selected_trials')  # Production Geometries
            if not isinstance(selected_trials, list):
                selected_trials = []
            if len(selected_trials) == 0:
                tprint_warning("Layer2 produced zero production geometries (no trials passed gates).")

            if events_df is not None and hasattr(events_df, "index"):
                evt_index = events_df.index
                try:
                    if isinstance(l2_labels, pd.Series):
                        l2_labels = l2_labels.reindex(evt_index)
                    if isinstance(l2_score, pd.Series):
                        l2_score = l2_score.reindex(evt_index)
                    if isinstance(l2_confidence, pd.Series):
                        l2_confidence = l2_confidence.reindex(evt_index)
                    if isinstance(l2_returns, pd.Series):
                        l2_returns = l2_returns.reindex(evt_index)
                    if isinstance(l2_weights, pd.Series):
                        l2_weights = l2_weights.reindex(evt_index)
                except Exception:
                    pass

                try:
                    if isinstance(l2_labels, pd.Series):
                        y = pd.to_numeric(l2_labels, errors="coerce").astype(float)
                        uniq = np.unique(y.dropna().values)
                        is_binary = bool(
                            (uniq.size <= 2)
                            and bool(np.all(np.isin(uniq, [0.0, 1.0])))
                        )
                        if not is_binary:
                            l2_labels = (y >= 0.5).astype(float).where(y.notna())
                except Exception:
                    pass

                try:
                    if isinstance(individual_geos, dict):
                        individual_geos = {
                            str(k): (v.reindex(evt_index) if isinstance(v, pd.Series) else v)
                            for k, v in individual_geos.items()
                        }
                except Exception:
                    pass

            # Save Layer 2 Production Geometries (Optimized on Full Data)
            with open(outcomes_dir / "layer2_selected_geometries.json", "w") as f:
                json.dump(selected_trials if isinstance(selected_trials, list) else [], f, indent=2, default=str)

            try:
                max_horizon = 0
                for t in list(selected_trials or []):
                    try:
                        if isinstance(t, dict):
                            params = t.get('params')
                            if isinstance(params, dict):
                                h = int(params.get('horizon', 0))
                                if h > max_horizon:
                                    max_horizon = h
                    except Exception:
                        continue
                if int(max_horizon) > 0:
                    tprint_info(f"Layer2 max production horizon={int(max_horizon)}")
                    config = dict(config)
                    purge_bars_eff = int(max_horizon) * 2
                    config.setdefault('layer3_max_lookahead_bars', int(purge_bars_eff))
                    config.setdefault('layer2_oof_purge_bars', int(purge_bars_eff))
                    config.setdefault('layer3_purge_bars', int(purge_bars_eff))
                    config.setdefault('layer3_embargo_bars', int(max(1, int(max_horizon))))
                    try:
                        tf_minutes = _timeframe_to_minutes(config.get("timeframe", "15m"))
                    except Exception:
                        tf_minutes = 15
                    config.setdefault('layer4_purge_minutes', int(max(60, int(tf_minutes) * int(purge_bars_eff))))
                    config.setdefault('layer4_embargo_minutes', int(max(30, int(tf_minutes) * int(max_horizon))))
            except Exception:
                pass

            # Persist a reusable bundle for layer3/layer4-only runs
            joblib.dump(
                {
                    "l2_labels": l2_labels,
                    "l2_score": l2_score,
                    "l2_confidence": l2_confidence,
                    "l2_returns": l2_returns,
                    "l2_weights": l2_weights,
                    "individual_geos": individual_geos,
                    "individual_variances": individual_variances, # Save Variances
                    "tree_stats": l2_output.get('tree_stats'), # Save Tree Stats for reporting
                    "events_df": events_df,
                    "selected_trials": selected_trials,
                    "l2_quality_weights": l2_quality_weights,
                },
                layer2_bundle_path,
            )

            if layer2_substep == "oof":
                tprint_success("Layer 2 OOF Bundle Saved.")
                return {"success": True}

            
        # ---------------------------------------------------------
        # Component Weights Preparation for Layer 3 Comparison
        # ---------------------------------------------------------
        
        # Try load weights from config or previous step if passed
        target_sample_weight = config.get('target_sample_weight')

        def _finalize_weight_series(w: pd.Series, name: str) -> pd.Series:
            s = pd.to_numeric(w, errors='coerce').astype(float)
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.fillna(1.0)
            s = s.clip(lower=0.0)
            try:
                m = float(s.mean())
            except Exception:
                m = 1.0
            if (not np.isfinite(m)) or m <= 0.0:
                tprint_warning(f"{name}: invalid mean weight; falling back to 1.0")
                return pd.Series(1.0, index=s.index)
            return s / m

        # Layer 1 Weights (must end aligned to events_df.index)
        w_l1_aligned: pd.Series
        if target_sample_weight is None:
            w_l1_aligned = pd.Series(1.0, index=events_df.index)
        elif isinstance(target_sample_weight, pd.Series):
            # Accept event-aligned weights (preferred)
            w_l1_aligned = target_sample_weight.reindex(events_df.index)
            if w_l1_aligned.isna().all():
                tprint_warning(
                    "Layer 1 target_sample_weight provided as Series but has no overlap with events_df.index; using 1.0"
                )
                w_l1_aligned = pd.Series(1.0, index=events_df.index)
            else:
                w_l1_aligned = _finalize_weight_series(w_l1_aligned, "layer1_weight")
        else:
            try:
                w_arr = np.asarray(target_sample_weight, dtype=float).reshape(-1)
            except Exception:
                w_arr = np.asarray([], dtype=float)

            if int(w_arr.size) == int(len(events_df.index)):
                w_l1_aligned = _finalize_weight_series(pd.Series(w_arr, index=events_df.index), "layer1_weight")
            elif int(w_arr.size) == int(len(market_data.index)):
                # bar-level weights -> reindex to events
                w_l1_series = pd.Series(w_arr, index=market_data.index)
                w_l1_aligned = _finalize_weight_series(
                    w_l1_series.reindex(events_df.index),
                    "layer1_weight",
                )
            else:
                tprint_warning(
                    "Layer 1 target_sample_weight length mismatch "
                    f"(got {int(w_arr.size)}; expected {len(events_df.index)} [events] or {len(market_data.index)} [bars]). Using 1.0."
                )
                w_l1_aligned = pd.Series(1.0, index=events_df.index)
        
        # Layer 2 Weights (Composite from Geometry Bagging)
        w_l2_aligned = l2_weights # Already aligned to events_df
        
        # Net Returns (for Magnitude)
        l2_returns_aligned = l2_returns # Already aligned to events_df
        
        # ---------------------------------------------------------
        # Data Assembly for Layer 3
        # ---------------------------------------------------------
        tprint_info(">>> Preparing OOF Data for Layer 3...")
        
        # Assemble OOF predictions from individual geometries
        geo_preds_df = pd.DataFrame(index=events_df.index)
        for uuid, preds in individual_geos.items():
            # preds are already Series on the correct index (or reindex safe)
            geo_preds_df[uuid] = preds.reindex(events_df.index)
            
        geo_cols = list(geo_preds_df.columns)
        
        l3_input_df = geo_preds_df.copy()

        # Inject Individual Variances for Layer 3 Uncertainty Features
        if individual_variances and isinstance(individual_variances, dict):
            for uuid, variances in individual_variances.items():
                var_col = f"{uuid}_var"
                if isinstance(variances, pd.Series):
                     l3_input_df[var_col] = variances.reindex(events_df.index)

        context_cols = ['volatility_1d', 'trend_regime', 'vol_regime']
        for c in context_cols:
            if c in events_df.columns:
                l3_input_df[c] = events_df[c]
            elif c in market_data.columns:
                 l3_input_df[c] = market_data.loc[l3_input_df.index, c]
        
        target_col = 'l2_consensus_target'
        l3_target = l2_labels
        try:
            if isinstance(l3_target, pd.Series):
                l3_target = l3_target.reindex(l3_input_df.index)
        except Exception:
            pass

        try:
            use_econ_target = bool(config.get("layer3_use_econ_target", True))
        except Exception:
            use_econ_target = True

        if use_econ_target:
            try:
                tx_cost = float(get_transaction_cost(config))
            except Exception:
                tx_cost = 0.0
            try:
                econ_mult = float(config.get("layer3_econ_win_tx_mult", 1.0))
            except Exception:
                econ_mult = 1.0
            if (not np.isfinite(econ_mult)) or econ_mult <= 0.0:
                econ_mult = 1.0

            try:
                deadband = config.get("layer3_econ_deadband")
                if isinstance(deadband, str) and deadband.strip().lower() == "auto":
                    deadband = None
                deadband = float(deadband) if deadband is not None else None
            except Exception:
                deadband = None

            try:
                deadband_mode = str(config.get("layer3_econ_deadband_mode", "auto"))
            except Exception:
                deadband_mode = "auto"

            if deadband is None and str(deadband_mode).strip().lower() == "auto":
                try:
                    cov_min = float(config.get("layer3_econ_coverage_min", 0.10))
                except Exception:
                    cov_min = 0.10
                try:
                    cov_max = float(config.get("layer3_econ_coverage_max", 0.30))
                except Exception:
                    cov_max = 0.30
                try:
                    grid_n = int(config.get("layer3_econ_deadband_grid", 41))
                except Exception:
                    grid_n = 41
                try:
                    min_cnt = int(config.get("layer3_econ_min_class_count", 200))
                except Exception:
                    min_cnt = 200

                try:
                    d_auto, d_diag = _select_econ_deadband_auto(
                        returns=l2_returns.reindex(l3_input_df.index),
                        coverage_min=float(cov_min),
                        coverage_max=float(cov_max),
                        grid_size=int(grid_n),
                        min_class_count=int(min_cnt),
                    )
                    deadband = float(d_auto)
                    tprint_info(
                        f"Layer3 econ deadband auto: deadband={deadband:.6g}, coverage={d_diag.get('coverage')}, "
                        f"pos_rate={d_diag.get('pos_rate')}, cohens_d={d_diag.get('cohens_d')}, snr={d_diag.get('snr')}"
                    )
                except Exception:
                    deadband = float(tx_cost) * float(econ_mult)

            if deadband is None:
                deadband = float(tx_cost) * float(econ_mult)
            if (not np.isfinite(deadband)) or deadband < 0.0:
                deadband = 0.0

            try:
                r = pd.to_numeric(l2_returns.reindex(l3_input_df.index), errors="coerce").astype(float)
            except Exception:
                r = pd.Series(np.nan, index=l3_input_df.index, dtype=float)

            try:
                econ_target_mode = str(config.get("layer3_econ_target_mode", "soft")).strip().lower()
            except Exception:
                econ_target_mode = "soft"

            if econ_target_mode in {"hard", "binary"}:
                econ_target = pd.Series(np.nan, index=l3_input_df.index, dtype=float)
                try:
                    econ_target.loc[r > float(deadband)] = 1.0
                    econ_target.loc[r < -float(deadband)] = 0.0
                except Exception:
                    pass
                l3_target = econ_target
            else:
                try:
                    soft_scale = float(config.get("layer3_econ_soft_scale", 2.0))
                except Exception:
                    soft_scale = 2.0
                if (not np.isfinite(soft_scale)) or soft_scale <= 0.0:
                    soft_scale = 2.0
                denom = float(deadband) if np.isfinite(deadband) and float(deadband) > 1e-12 else 1e-3

                # Robust clipping for returns (prevent extreme outliers from skewing sigmoid)
                # Clip returns to [P01, P99] or MAD-based bounds if needed.
                # Here we clip z-score directly to [-5, 5] (more conservative than -10, 10)
                # to ensure we don't saturate too easily, while still allowing strong signals.
                z = pd.to_numeric(r / float(denom), errors="coerce").astype(float)
                z = z.clip(lower=-5.0, upper=5.0)

                econ_soft = 1.0 / (1.0 + np.exp(-float(soft_scale) * z))
                econ_soft = econ_soft.where(r.notna())
                l3_target = econ_soft

        try:
            target_all_nan = (not isinstance(l3_target, pd.Series)) or bool(pd.to_numeric(l3_target, errors="coerce").isna().all())
        except Exception:
            target_all_nan = True

        if target_all_nan:
            try:
                geo_mean = geo_preds_df.mean(axis=1, skipna=True)
                geo_count = geo_preds_df.notna().sum(axis=1)
                fallback_target = (geo_mean >= 0.5).astype(float)
                fallback_target.loc[geo_count <= 0] = np.nan
                l3_target = fallback_target
            except Exception:
                l3_target = pd.Series(np.nan, index=l3_input_df.index, dtype=float)

        l3_input_df[target_col] = l3_target

        try:
            ts_run = str(config.get("run_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))
        except Exception:
            ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            drop_rows = []
            n_in = int(len(l3_input_df))
            ret_s = pd.to_numeric(l2_returns.reindex(l3_input_df.index), errors="coerce")
            tgt_s = pd.to_numeric(l3_target.reindex(l3_input_df.index), errors="coerce") if isinstance(l3_target, pd.Series) else pd.Series(np.nan, index=l3_input_df.index, dtype=float)
            labeled_mask = tgt_s.notna()

            try:
                geo_cols_local = list(geo_cols or [])
                if geo_cols_local:
                    geo_non_missing = l3_input_df[geo_cols_local].notna().sum(axis=1)
                    all_geo_missing = geo_non_missing <= 0
                else:
                    all_geo_missing = pd.Series(False, index=l3_input_df.index)
            except Exception:
                all_geo_missing = pd.Series(False, index=l3_input_df.index)

            drop_rows.append({"metric": "n_rows_input", "count": n_in, "pct": 1.0})
            drop_rows.append({"metric": "realized_return_nan", "count": int(ret_s.isna().sum()), "pct": float(ret_s.isna().mean()) if n_in > 0 else float('nan')})
            drop_rows.append({"metric": "target_nan", "count": int(tgt_s.isna().sum()), "pct": float(tgt_s.isna().mean()) if n_in > 0 else float('nan')})
            drop_rows.append({"metric": "filtered_by_labeled_mask", "count": int((~labeled_mask).sum()), "pct": float((~labeled_mask).mean()) if n_in > 0 else float('nan')})
            drop_rows.append({"metric": "all_geo_preds_missing", "count": int(all_geo_missing.sum()), "pct": float(all_geo_missing.mean()) if n_in > 0 else float('nan')})

            for c in ['volatility_1d', 'trend_regime', 'vol_regime']:
                if c in l3_input_df.columns:
                    s = pd.Series(l3_input_df[c], index=l3_input_df.index)
                    miss = s.isna()
                    drop_rows.append({"metric": f"{c}_missing", "count": int(miss.sum()), "pct": float(miss.mean()) if n_in > 0 else float('nan')})
            out_drop = outcomes_dir / f"layer3_drop_reasons_{ts_run}.csv"
            pd.DataFrame(drop_rows).to_csv(out_drop, index=False)
            tprint_info(f"Layer3 drop reasons written: {out_drop}")
        except Exception:
            pass

        if start_at == "layer4" and not bool(config.get('force_hpo', False)):
            layer3_oof_path = outcomes_dir / "layer3_oof_preds.csv"
            if not layer3_oof_path.exists():
                raise FileNotFoundError(
                    f"Missing Layer3 OOF predictions at {layer3_oof_path}. Run with --labeling-hpo-start-at layer3 first."
                )
            oof_export = _read_layer3_oof_csv(layer3_oof_path)
            final_model = None
        else:
            # ---------------------------------------------------------
            # LAYER 3: Calibration & Meta-Model (OOF & Final)
            # ---------------------------------------------------------
            tprint_info(">>> Executing Layer 3: Weighting Scheme Comparison & Training...")

            # --- Target Variant 1: Economic Target (Default) ---
            tprint_info("   [Target A] Training on Economic Target (continuous)...")
            l3_input_econ = l3_input_df.copy()
            l3_input_econ[target_col] = l3_target  # Economic target (soft)

            oof_export_econ, final_model_econ = layer3_analyst_lgbm(
                oof_df=l3_input_econ,
                base_model_cols=geo_cols,
                target_col=target_col,
                train_split_date=None,
                layer1_weight=w_l1_aligned,
                layer2_weight=w_l2_aligned,
                layer2_weight_quality=l2_quality_weights,
                net_returns=l2_returns_aligned,
                market_data=market_data,
                config=config,
            )
            
            # --- Target Variant 2: Binary Consensus (Layer 2) ---
            tprint_info("   [Target B] Training on Binary Consensus (Layer 2)...")
            l3_input_bin = l3_input_df.copy()
            # Restore binary labels from Layer 2 (l2_labels is binary thresholded at 0.5)
            l2_labels_binary = l2_labels.reindex(l3_input_bin.index)
            l3_input_bin[target_col] = l2_labels_binary

            oof_export_bin, final_model_bin = layer3_analyst_lgbm(
                oof_df=l3_input_bin,
                base_model_cols=geo_cols,
                target_col=target_col,
                train_split_date=None,
                layer1_weight=w_l1_aligned,
                layer2_weight=w_l2_aligned,
                layer2_weight_quality=l2_quality_weights,
                net_returns=l2_returns_aligned,
                market_data=market_data,
                config=config,
            )

            # --- Compare and Select Winner ---
            # We compare ScoreL3 from the 'Winner' scheme of each run.
            # We need to extract the score from the report data or recompute.
            # Since layer3_analyst_lgbm returns (df, model) and prints/saves report internally,
            # we need to look at the summary CSVs it generated or calculate metrics now.

            # Helper to get score from OOF
            def _quick_score(df_oof, t_col):
                met = _compute_layer3_metrics(df_oof, target_col=t_col, prob_col="meta_prob")
                if "error" in met: return -999.0
                # ScoreL3 approximation
                auc = met.get("auc", 0.5)
                ll = met.get("log_loss", 0.693)
                ece = met.get("ece", 0.0)
                return 100 * (auc - 0.5) + 50 * (0.693 - ll) - 200 * ece

            # Note: For Economic Target run, target_col has soft labels.
            # For Binary Target run, target_col has binary labels.
            # Metrics are comparable if we evaluate against the specific target used for training?
            # Or should we evaluate both against the Economic outcome?
            # Usually we select based on the target it was trained for (internal consistency).

            score_econ = _quick_score(oof_export_econ, target_col)
            score_bin = _quick_score(oof_export_bin, target_col) # uses binary labels in this DF

            tprint_info(f"   [Comparison] Economic Target Score: {score_econ:.4f}")
            tprint_info(f"   [Comparison] Binary Target Score:   {score_bin:.4f}")

            # Create Comparison Table
            comp_rows = [
                {"Target_Type": "Economic (Soft)", "Score": score_econ, "Used_For_Production": False},
                {"Target_Type": "Binary (L2)",     "Score": score_bin,  "Used_For_Production": False}
            ]

            # Logic: Default to Economic unless Binary is significantly better?
            # Or trust the user preference?
            # Current default is Economic.
            # If user explicitly wants binary, they set layer3_use_econ_target=False.
            # But here we force-ran both.
            # Let's keep Economic as primary unless configured otherwise, or if Binary is strictly better.
            # Actually, standardizing on Economic is the design intent. We'll keep Econ as production
            # but export both metrics.

            # Allow config override to pick binary
            force_binary = not bool(config.get("layer3_use_econ_target", True))

            if force_binary:
                tprint_info("   [Selection] Config forced Binary Target.")
                oof_export = oof_export_bin
                final_model = final_model_bin
                comp_rows[1]["Used_For_Production"] = True
            else:
                tprint_info("   [Selection] Defaulting to Economic Target.")
                oof_export = oof_export_econ
                final_model = final_model_econ
                comp_rows[0]["Used_For_Production"] = True

            # Save Comparison Table
            try:
                comp_df = pd.DataFrame(comp_rows)
                comp_path = outcomes_dir / f"layer3_target_comparison_{ts_run}.csv"
                comp_df.to_csv(comp_path, index=False)

                # Append to Report
                md_path = outcomes_dir / f"layer3_report_{config.get('symbol')}_{config.get('timeframe')}_{ts_run}.md"
                if md_path.exists():
                    with open(md_path, "a") as f:
                        f.write("\n\n## Target Definition Comparison\n")
                        f.write(comp_df.to_markdown(index=False))
                        f.write("\n")
            except Exception:
                pass
            
            # Calculate final composite weight for artifact saving (using Scheme 7 logic as default/reference)
            # Note: The actual model training inside layer3 uses the BEST scheme found.
            # But for 'weights_stats.csv', we save the reference composite one.
            magnitude_factor = np.log1p(l2_returns.abs().fillna(0))
            w_final_series = w_l2_aligned * magnitude_factor * w_l1_aligned
            if w_final_series.mean() > 0:
                w_final_series /= w_final_series.mean()
            w_final = w_final_series.values

            # Generate Diagnostics (on OOF predictions)
            tprint_info(">>> Generating Layer 3 Diagnostics...")
            plot_diagnostics(
                y_true=oof_export[target_col],
                y_prob=oof_export['meta_prob'],
                output_path=str(outcomes_dir / "layer3_calibration_plot.png")
            )
            
            # Save OOF Predictions (Full History)
            layer3_oof_path = outcomes_dir / "layer3_oof_preds.csv"
            oof_export.to_csv(layer3_oof_path)
            
            # Save Weights
            pd.DataFrame({'weight': w_final}).describe().to_csv(outcomes_dir / "layer3_weights_stats.csv")
            
            # Save Final Model
            joblib.dump(final_model, outcomes_dir / "layer3_final_model.joblib")


        # ---------------------------------------------------------
        # LAYER 4: Risk Filter (ExtraTrees + Platt Calibration)
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 4: Risk Filter (ExtraTrees + Platt Calibration)...")

        # Prepare Data for Layer 4 Risk Filter
        l4_input = oof_export.copy()

        try:
            if 'symbol' not in l4_input.columns:
                l4_input['symbol'] = str(config.get('symbol', ''))
            if 'timeframe' not in l4_input.columns:
                l4_input['timeframe'] = str(config.get('timeframe', ''))
        except Exception:
            pass

        # Attach realized returns if not present (from Layer 2 OOF returns)
        if 'realized_return' not in l4_input.columns:
            l4_input['realized_return'] = l2_returns.reindex(l4_input.index).fillna(0.0)

        # Attach volatility if not present
        if 'volatility_1d' not in l4_input.columns and 'volatility_1d' in l3_input_df.columns:
            l4_input['volatility_1d'] = l3_input_df['volatility_1d']

        # Layer 4 Risk Filter Configuration
        layer4_enabled = bool(config.get('layer4_risk_filter_enabled', True))
        layer4_quantile_threshold = float(config.get('layer4_quantile_threshold', 0.6))
        layer4_final_score_formula = str(config.get('layer4_final_score_formula', 'dynamic'))

        p_col_for_sizing = 'meta_prob'

        l4_risk_metrics = {}
        l4_final_probs = l4_input['meta_prob'].copy()  # Default: use L3 probs directly

        if layer4_enabled and market_data is not None:
            try:
                tprint_info("   Training Layer 4 Risk Filter OOF (no leakage)...")

                try:
                    l4_oof_folds = int(config.get('layer4_oof_folds', 5))
                except Exception:
                    l4_oof_folds = 5

                try:
                    grid_thresholds = config.get('layer4_grid_thresholds', [0.7, 0.6, 0.5, 0.4, 0.3])
                    l3_grid_thresholds = [float(x) for x in (grid_thresholds or [])]
                    if not l3_grid_thresholds:
                        l3_grid_thresholds = [0.7, 0.6, 0.5, 0.4, 0.3]
                except Exception:
                    l3_grid_thresholds = [0.7, 0.6, 0.5, 0.4, 0.3]

                l4_oof_df, l4_oof_metrics = train_layer4_oof(
                    oof_df=l4_input,
                    market_data=market_data,
                    l3_prob_col='meta_prob',
                    target_col=target_col,
                    return_col='realized_return',
                    l3_quantile_thresholds=l3_grid_thresholds,
                    n_folds=l4_oof_folds,
                    config=config,
                )

                if isinstance(l4_oof_df, pd.DataFrame) and (not l4_oof_df.empty) and ('layer4_prob' in l4_oof_df.columns):
                    l4_input = l4_oof_df.copy()
                    l4_probs = pd.to_numeric(l4_input['layer4_prob'], errors='coerce').to_numpy(dtype=float, copy=False)
                    # Use L4 gate score directly as final score (gate is authoritative)
                    final_scores = l4_probs

                    l4_input['final_score'] = final_scores
                    l4_final_probs = pd.Series(final_scores, index=l4_input.index)
                    p_col_for_sizing = 'final_score'

                    # Train a final model on the full data for saving (evaluation uses OOF above)
                    try:
                        l4_regime_features = compute_layer4_regime_features(market_data)
                        common_idx = l4_input.index.intersection(l4_regime_features.index)
                        if len(common_idx) >= 100:
                            X_full = l4_regime_features.loc[common_idx]
                            y_full_raw = pd.to_numeric(l4_input.loc[common_idx, target_col], errors='coerce')
                            l3_full = pd.to_numeric(l4_input.loc[common_idx, 'meta_prob'], errors='coerce')

                            labeled = y_full_raw.notna() & pd.to_numeric(l3_full, errors='coerce').notna()
                            if int(labeled.sum()) >= 100:
                                try:
                                    include_l3_prob_feature = bool(config.get('layer4_include_l3_prob_feature', True))
                                except Exception:
                                    include_l3_prob_feature = True

                                # Calculate meta-features on full series before slicing
                                if include_l3_prob_feature:
                                    X_full = X_full.copy()
                                    l3_full_numeric = pd.to_numeric(l3_full, errors='coerce')
                                    X_full['l3_prob'] = l3_full_numeric
                                    X_full['l3_lag'] = l3_full_numeric.ewm(span=5, adjust=False).mean()

                                X_fit = X_full.loc[labeled]
                                y_fit = (pd.to_numeric(y_full_raw.loc[labeled], errors='coerce').astype(float) >= 0.5).astype(int)
                                l3_fit = pd.to_numeric(l3_full.loc[labeled], errors='coerce')

                                l4_risk_filter = Layer4RiskFilter(
                                    n_estimators=int(config.get('layer4_n_estimators', 100)),
                                    max_depth=int(config.get('layer4_max_depth', 5)),
                                    min_samples_leaf=int(config.get('layer4_min_samples_leaf', 20)),
                                    l3_quantile_threshold=layer4_quantile_threshold,
                                )
                                l4_risk_filter.fit(X=X_fit, y_true=y_fit, l3_probs=l3_fit)
                                if l4_risk_filter._is_fitted:
                                    l4_model_path = outcomes_dir / "layer4_risk_filter.joblib"
                                    l4_risk_filter.save(str(l4_model_path))
                    except Exception:
                        pass

                    l4_risk_metrics = {
                        'layer4_enabled': True,
                        'layer4_quantile_threshold': layer4_quantile_threshold,
                        'layer4_final_score_formula': layer4_final_score_formula,
                        'layer4_n_samples': int(l4_input.shape[0]),
                        'layer4_prob_mean': float(np.nanmean(l4_probs)),
                        'layer4_prob_std': float(np.nanstd(l4_probs)),
                        'final_score_mean': float(np.nanmean(final_scores)),
                        'final_score_std': float(np.nanstd(final_scores)),
                        'oof': l4_oof_metrics,
                    }

                    tprint_success(f"   Layer 4 Risk Filter OOF complete. Final score formula: {layer4_final_score_formula}")
                else:
                    tprint_warning("   Layer 4 Risk Filter OOF failed/empty. Using L3 probs directly.")
                    l4_risk_metrics = {'layer4_enabled': False, 'reason': 'oof_failed_or_empty'}
            except Exception as e:
                tprint_warning(f"   Layer 4 Risk Filter error: {e}. Using L3 probs directly.")
                l4_risk_metrics = {'layer4_enabled': False, 'reason': str(e)}
        else:
            l4_risk_metrics = {'layer4_enabled': False, 'reason': 'disabled_or_no_market_data'}

        # Save Layer 4 metrics
        with open(outcomes_dir / "layer4_risk_filter_metrics.json", "w") as f:
            json.dump(l4_risk_metrics, f, indent=2, default=str)

        # ---------------------------------------------------------
        # LAYER 5: Position Sizing & Portfolio Diagnostics
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 5: Position Sizing & Portfolio Diagnostics...")

        # Use final_score (Layer4 OOF) if available, otherwise meta_prob
        if 'final_score' in l4_input.columns:
            p_col_for_sizing = 'final_score'

        # Initialize Sizer
        try:
            l4_tx_cost = float(config.get('layer4_transaction_cost', 0.0))
        except Exception:
            l4_tx_cost = 0.0

        try:
            # 1. Start with config default
            layer4_p_min = float(config.get('layer4_p_min', 0.5))
            
            # 2. Dynamic Override: Use best threshold from Layer 4 OOF Grid if available
            if isinstance(l4_risk_metrics, dict) and 'oof' in l4_risk_metrics:
                best_cfg = l4_risk_metrics['oof'].get('best')
                if best_cfg and 'l3_threshold' in best_cfg:
                    best_thresh = float(best_cfg['l3_threshold'])
                    if np.isfinite(best_thresh) and best_thresh > 1e-6:
                        tprint_info(f"   [DYNAMIC_THRESHOLD] Overriding p_min {layer4_p_min:.4f} -> {best_thresh:.4f} (best l3_threshold from grid)")
                        layer4_p_min = best_thresh
                        
        except Exception as e:
            tprint_warning(f"   Could not determine dynamic p_min: {e}. Using default 0.5")
            layer4_p_min = 0.5

        try:
            layer4_p_max = float(config.get('layer4_p_max', 0.9))
        except Exception:
            layer4_p_max = 0.9

        try:
            layer4_gate_mode = str(config.get('layer4_gate_mode', 'top_k_per_day'))
        except Exception:
            layer4_gate_mode = 'top_k_per_day'
        try:
            layer4_gate_quantile = config.get('layer4_gate_quantile')
            layer4_gate_quantile = float(layer4_gate_quantile) if layer4_gate_quantile is not None else None
        except Exception:
            layer4_gate_quantile = None
        try:
            layer4_gate_top_k = config.get('layer4_gate_top_k')
            layer4_gate_top_k = int(layer4_gate_top_k) if layer4_gate_top_k is not None else None
        except Exception:
            layer4_gate_top_k = None
        try:
            layer4_gate_top_k_per_day = config.get('layer4_gate_top_k_per_day')
            layer4_gate_top_k_per_day = int(layer4_gate_top_k_per_day) if layer4_gate_top_k_per_day is not None else None
        except Exception:
            layer4_gate_top_k_per_day = None

        if str(layer4_gate_mode).strip().lower() == 'top_k_per_day' and (layer4_gate_top_k_per_day is None or int(layer4_gate_top_k_per_day) <= 0):
            layer4_gate_top_k_per_day = 2

        try:
            layer4_gate_search_q_low = config.get('layer4_gate_search_q_low')
            layer4_gate_search_q_low = float(layer4_gate_search_q_low) if layer4_gate_search_q_low is not None else None
        except Exception:
            layer4_gate_search_q_low = None
        try:
            layer4_gate_search_q_high = config.get('layer4_gate_search_q_high')
            layer4_gate_search_q_high = float(layer4_gate_search_q_high) if layer4_gate_search_q_high is not None else None
        except Exception:
            layer4_gate_search_q_high = None
        try:
            layer4_gate_search_min_range = config.get('layer4_gate_search_min_range')
            layer4_gate_search_min_range = float(layer4_gate_search_min_range) if layer4_gate_search_min_range is not None else None
        except Exception:
            layer4_gate_search_min_range = None
        try:
            layer4_gate_search_max_iter = config.get('layer4_gate_search_max_iter')
            layer4_gate_search_max_iter = int(layer4_gate_search_max_iter) if layer4_gate_search_max_iter is not None else None
        except Exception:
            layer4_gate_search_max_iter = None

        sizer = Layer5PositionSizer(
            oof_df=l4_input,
            p_col=p_col_for_sizing,  # Use final_score from Layer 4 if available, else meta_prob
            target_col=target_col,
            return_col='realized_return',
            transaction_cost=0.0, # CRITICAL: Layer 2 returns are ALREADY Net of costs. Do not double count.
            gamma=1.2,
            p_min=float(layer4_p_min),
            p_max=float(layer4_p_max),
            gate_mode=str(layer4_gate_mode),
            gate_quantile=layer4_gate_quantile,
            gate_top_k=layer4_gate_top_k,
            gate_top_k_per_day=layer4_gate_top_k_per_day,
            gate_search_q_low=layer4_gate_search_q_low,
            gate_search_q_high=layer4_gate_search_q_high,
            gate_search_min_range=layer4_gate_search_min_range,
            gate_search_max_iter=layer4_gate_search_max_iter,
        )

        # Run Backtest
        l4_metrics = sizer.run_backtest()

        traded_idx = None
        try:
            if isinstance(getattr(sizer, "df", None), pd.DataFrame) and "layer5_size" in sizer.df.columns:
                traded_mask = pd.to_numeric(sizer.df["layer5_size"], errors="coerce").astype(float) > 1e-4
                traded_idx = sizer.df.index[traded_mask]
        except Exception:
            traded_idx = None

        # Save Layer 5 Artifacts (already called internally by run_backtest)
        # sizer._save_artifacts_to_disk(l4_metrics)  # Removed: run_backtest() calls this internally

        # Helper to sanitize keys for JSON (e.g. Intervals)
        def sanitize_keys(d):
            if isinstance(d, dict):
                return {str(k): sanitize_keys(v) for k, v in d.items()}
            if isinstance(d, list):
                return [sanitize_keys(x) for x in d]
            return d

        l4_metrics_safe = sanitize_keys(l4_metrics)



        with open(outcomes_dir / "layer5_performance_metrics.json", "w") as f:
            json.dump(l4_metrics_safe, f, indent=2, default=str)

        tprint_success(f"Layer 5 Completed. Metrics: {json.dumps(l4_metrics_safe, indent=2, default=str)}")

        tprint_success(f"Pipeline Completed. Artifacts saved to {outcomes_dir}")
        
        layer2_metrics = _compute_layer2_metrics(l2_labels, l2_returns, l2_weights)
        layer3_metrics = _compute_layer3_metrics(oof_export, target_col=target_col, prob_col="meta_prob")
        try:
            mp = pd.to_numeric(oof_export.get('meta_prob'), errors='coerce') if isinstance(oof_export, pd.DataFrame) else None
            if mp is not None:
                layer3_metrics['meta_prob_nan'] = int(mp.isna().sum())
                layer3_metrics['meta_prob_nan_pct'] = float(mp.isna().mean()) if int(len(mp)) > 0 else float('nan')
        except Exception:
            pass
        try:
            gate_idx = None
            try:
                gate_idx = sizer.get_gate_index()
            except Exception:
                gate_idx = None

            if gate_idx is not None:
                gate_idx = pd.Index(gate_idx).intersection(oof_export.index)
                layer3_metrics["gate_n"] = int(len(gate_idx))
                if int(len(gate_idx)) > 0:
                    l3_gate = _compute_layer3_metrics(oof_export.loc[gate_idx], target_col=target_col, prob_col="meta_prob")
                    if isinstance(l3_gate, dict):
                        for k, v in l3_gate.items():
                            layer3_metrics[f"gate_{k}"] = v

            if traded_idx is not None:
                traded_idx = pd.Index(traded_idx).intersection(oof_export.index)
                layer3_metrics["size_trade_n"] = int(len(traded_idx))
                if int(len(traded_idx)) > 0:
                    l3_traded = _compute_layer3_metrics(oof_export.loc[traded_idx], target_col=target_col, prob_col="meta_prob")
                    if isinstance(l3_traded, dict):
                        for k, v in l3_traded.items():
                            layer3_metrics[f"size_trade_{k}"] = v
        except Exception:
            pass

        try:
            p_max = float(getattr(sizer, "p_max", 0.9))
        except Exception:
            p_max = 0.9
        try:
            gamma = float(getattr(sizer, "gamma", 1.2))
        except Exception:
            gamma = 1.2

        sweep_path = _write_layer5_pmin_sweep(
            outcomes_dir=outcomes_dir,
            l4_input=l4_input,
            target_col=target_col,
            p_col=p_col_for_sizing,
            return_col="realized_return",
            p_max=p_max,
            gamma=gamma,
            transaction_cost=0.0,
            p_min_values=None,
        )
        unified_paths = _write_unified_label_based_report(
            outcomes_dir=outcomes_dir,
            context={
                "symbol": config.get("symbol"),
                "exchange": config.get("exchange"),
                "timeframe": config.get("timeframe"),
                "direction": config.get("direction"),
                "outcomes_dir": str(outcomes_dir),
            },
            layer2_metrics=layer2_metrics,
            layer3_metrics=layer3_metrics,
            layer4_risk_metrics=l4_risk_metrics,
            layer5_metrics=dict(l4_metrics_safe, **({"pmin_sweep_csv": sweep_path} if sweep_path else {})),
        )
        
        # ---------------------------------------------------------------------
        # OOS TEST: Run OOS evaluation if enabled
        # ---------------------------------------------------------------------
        if oos_data is not None and oos_config.get('enabled', False):
            tprint_info(">>> Running OOS Layer 5 Evaluation...")
            try:
                # Generate OOS predictions using trained models
                trained_models = {
                    'layer2': layer2,
                    'layer3': final_model,
                    'layer4': l4_model if 'l4_model' in locals() else None
                }
                
                oos_predictions = generate_oos_predictions(
                    oos_data=oos_data,
                    trained_models=trained_models,
                    config=config
                )
                
                # Run Layer 5 on OOS predictions
                oos_layer5_results = run_oos_layer5_evaluation(
                    oos_predictions=oos_predictions,
                    config=config
                )
                
                # Compare with OOF results
                oof_layer5_results = l4_metrics_safe
                oos_comparison = compare_oos_vs_oof(
                    oos_results=oos_layer5_results,
                    oof_results=oof_layer5_results,
                    config=config
                )
                
                # Assemble OOS results
                oos_results = {
                    'enabled': True,
                    'split_date': split_date,
                    'train_bars': len(train_data),
                    'oos_bars': len(oos_data),
                    'layer5_metrics': oos_layer5_results,
                    'comparison': oos_comparison,
                    'reliability_score': oos_comparison.get('reliability_score', 0)
                }
                
                # Save OOS results
                save_oos_results(
                    results=oos_results,
                    config=config,
                    symbol=config.get('symbol', 'UNKNOWN'),
                    timeframe=config.get('timeframe', 'UNKNOWN')
                )
                
                tprint_success(f"OOS Test Complete. Reliability Score: {oos_comparison.get('reliability_score', 0):.0f}/100")
                
            except Exception as e:
                tprint_warning(f"OOS evaluation failed: {e}")
                oos_results = {'enabled': True, 'error': str(e)}
        else:
            oos_results = {'enabled': False}

        snr_reports: Dict[str, Any] = {}
        run_snr = bool(config.get("run_snr_diagnostics_after_layer4", True))
        if run_snr:
            try:
                from src.training.steps.labeling.snr_diagnostics import run_full

                def _latest_report(prefix: str) -> Optional[str]:
                    try:
                        matches = sorted(
                            outcomes_dir.glob(f"{prefix}_{config.get('symbol')}_{config.get('timeframe')}_*.json"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if matches:
                            return str(matches[0])
                    except Exception:
                        pass
                    return None

                def _run_for_direction(dir_value: str) -> None:
                    run_full(
                        symbol=str(config.get("symbol", "")),
                        exchange=str(config.get("exchange", "")),
                        timeframe=str(config.get("timeframe", "")),
                        direction=dir_value,
                        model=str(config.get("model", "analyst")),
                        cv_splits_learn=int(config.get("snr_cv_splits_learn", 3)),
                        cv_splits_robust=int(config.get("snr_cv_splits_robust", 5)),
                        prob_column=str(config.get("snr_prob_column", "meta_probability")),
                        prob_thresholds=config.get("snr_prob_thresholds"),
                        outcomes_dir=outcomes_dir,
                    )

                    snr_reports_key = f"snr_full_diagnostics_{dir_value}"
                    snr_reports[snr_reports_key] = _latest_report("snr_full_diagnostics")

                dir_cfg = str(config.get("direction", "long")).lower()
                if dir_cfg == "both":
                    _run_for_direction("long")
                    _run_for_direction("short")
                else:
                    _run_for_direction(dir_cfg)
            except Exception as exc:
                tprint_warning(f"SNR diagnostics failed after Layer4: {exc}")

        artifacts: Dict[str, Any] = {
            "layer2_geometries": str(outcomes_dir / "layer2_selected_geometries.json"),
            "layer5_events": str(outcomes_dir / "layer5_sized_events.csv"),
            "layer5_metrics": str(outcomes_dir / "layer5_performance_metrics.json"),
            "unified_report_csv": unified_paths["csv"],
            "unified_report_json": unified_paths["json"],
            "snr_diagnostics": snr_reports,
            "oos_results": oos_results,
        }

        try:
            artifacts["oof_preds"] = str(layer3_oof_path)
        except Exception:
            pass

        layer3_plot = outcomes_dir / "layer3_calibration_plot.png"
        if layer3_plot.exists():
            artifacts["calibration_plot"] = str(layer3_plot)

        layer3_model = outcomes_dir / "layer3_final_model.joblib"
        if layer3_model.exists():
            artifacts["final_model"] = str(layer3_model)

        return {
            "success": True,
            "outcomes_dir": str(outcomes_dir),
            "metrics": {
                "n_events": len(l3_input_df),
                "n_geometries": len(geo_cols),
                **l4_metrics
            },
            "artifacts": artifacts,
        }

def register_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry
    step_registry.register("meta_labeling_hpo_sample_weighted", MetaLabelingHPOSampleWeightedStep)
    # Aliases
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb_weighted", MetaLabelingHPOSampleWeightedStep)

register_meta_labeling_hpo_sample_weighted_step()
