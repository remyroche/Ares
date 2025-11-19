#!/usr/bin/env python3
"""SNR & Label Diagnostics for Meta-Labeling Outputs.

Usage examples (from project root):

  python scripts/snr_diagnostics.py label-quality \
      --symbol ETHUSDT --exchange binance --timeframe 15m

  python scripts/snr_diagnostics.py label-learnability \
      --symbol ETHUSDT --exchange binance --timeframe 15m

  python scripts/snr_diagnostics.py model-robustness \
      --symbol ETHUSDT --exchange binance --timeframe 15m

Subcommands:
- label-quality:     Label distribution, coverage, economic SNR, retention.
- label-learnability:Learnability (AUC-based) and entropy/balance of labels.
- model-robustness:  Probe model CV stability (AUC mean/std across folds).

This script is designed to be run *after* the
`feature_generation_meta_labeling_step` has been executed via the launcher,
so that the `labeled_data_{symbol}_{timeframe}` artifact exists.
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger  # type: ignore
from src.training.steps.labeling.feature_generation_meta_labeling_step import (  # type: ignore
    FeatureGenerationMetaLabelingStep,
    compute_learnability_score,
    compute_label_entropy_score,
    combined_label_quality_objective,
    DEFAULT_TRANSACTION_COST,
)

try:
    import lightgbm as lgb  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "snr_diagnostics requires lightgbm and scikit-learn to be installed. "
        "Install them in your environment before running this script."
    ) from exc


logger = system_logger.getChild("snr_diagnostics")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


OUTCOMES_DIR = Path("outcomes")


_LAST_EXPORTS: dict[str, dict] = {}


def _ensure_outcomes_dir() -> Path:
    """Ensure outcomes directory exists and return it."""
    OUTCOMES_DIR.mkdir(exist_ok=True)
    return OUTCOMES_DIR


def _export_report(
    prefix: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    payload: dict,
    markdown_lines: list[str],
) -> tuple[Path, Path]:
    """Export diagnostics payload as JSON and Markdown into outcomes/.

    Filenames are of the form:
        outcomes/{prefix}_{symbol}_{timeframe}_{YYYYMMDD_HHMMSS}.json/md
    """
    out_dir = _ensure_outcomes_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{symbol}_{timeframe}_{ts}"

    # Enrich payload with common metadata
    meta = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "model": model,
        "prefix": prefix,
        "timestamp_utc": ts,
    }
    full_payload = {"metadata": meta, **payload}

    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"

    with json_path.open("w") as f_json:
        json.dump(full_payload, f_json, indent=2, default=str)

    with md_path.open("w") as f_md:
        f_md.write("\n".join(markdown_lines))

    _LAST_EXPORTS[prefix] = {
        "json_path": json_path,
        "md_path": md_path,
        "payload": full_payload,
        "markdown_lines": markdown_lines,
    }

    logger.info("Saved %s diagnostics to %s and %s", prefix, json_path, md_path)
    return json_path, md_path


def _load_labeled_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
) -> pd.DataFrame:
    """Load labeled_data artifact produced by FeatureGenerationMetaLabelingStep.

    Tries both versioned HDF5 and legacy artifacts via the same BaseStep
    `_get_artifact` mechanism, so it remains compatible with older runs.
    """
    step = FeatureGenerationMetaLabelingStep()
    step.set_context(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
    )

    # Primary artifact name used by the meta-labeling step
    primary_name = f"labeled_data_{symbol}_{timeframe}"

    # Legacy/candidate names for compatibility
    candidate_names = [
        primary_name,
        f"labeled_data_{symbol}_{exchange}_{timeframe}",
        f"labeled_data_{symbol}_{timeframe}_{direction}",
    ]

    for name in candidate_names:
        try:
            df = step._get_artifact(  # type: ignore[attr-defined]
                artifact_name=name,
                artifact_type="data",
                data_category="features",
            )
        except Exception:
            df = None

        if isinstance(df, pd.DataFrame) and not df.empty:
            logger.info(
                "Loaded labeled data from artifact '%s' with shape %s",
                name,
                df.shape,
            )
            return df

    raise FileNotFoundError(
        f"Could not locate labeled_data artifact for {symbol} {exchange} {timeframe}. "
        f"Tried names: {candidate_names}. Run feature_generation_meta_labeling_step first."
    )


def _build_feature_matrix_from_labeled(labeled_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct (X, y) for learnability / robustness diagnostics from labeled_data.

    - y: uses `binary_label` column.
    - X: all numeric columns except obvious target/label/return-related ones.
    """
    if "binary_label" not in labeled_df.columns:
        raise ValueError("labeled_data is missing required 'binary_label' column")

    y = labeled_df["binary_label"].copy()

    # Numeric feature candidates
    numeric = labeled_df.select_dtypes(include=[np.number]).copy()

    # Drop columns that are clearly targets/labels/returns or sample weights
    drop_patterns = [
        "target",
        "label",
        "return",
        "meta_probability",
        "r_multiple",
        "sample_weight",
        "event_duration",
        "adaptive_profit_threshold",
        "adaptive_stop_threshold",
    ]
    drop_cols = []
    for col in numeric.columns:
        lower = col.lower()
        if any(pat in lower for pat in drop_patterns):
            drop_cols.append(col)

    X = numeric.drop(columns=drop_cols, errors="ignore")

    # Align X and y on common index and drop NaNs in y
    valid_mask = ~y.isna()
    y_clean = y[valid_mask]
    X_clean = X.loc[y_clean.index].fillna(0)

    if len(y_clean) < 50:
        logger.warning("Only %d valid samples after cleaning; diagnostics may be noisy", len(y_clean))

    return X_clean, y_clean


# --------------------------------------------------------------------------------------
# Label-quality diagnostics
# --------------------------------------------------------------------------------------


def run_label_quality(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
) -> None:
    """Compute label-quality and economic SNR diagnostics from labeled_data."""
    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)

    if "binary_label" not in df.columns or "realized_return" not in df.columns:
        raise ValueError("labeled_data must contain 'binary_label' and 'realized_return' columns")

    binary_labels = df["binary_label"]
    realized_returns = df["realized_return"]

    n_samples = len(df)
    labeled_mask = ~binary_labels.isna()
    n_labeled = int(labeled_mask.sum())

    n_positive = int((binary_labels == 1.0).sum())
    n_negative = int((binary_labels == 0.0).sum())
    positive_rate = n_positive / n_labeled if n_labeled > 0 else 0.0
    coverage = n_labeled / n_samples if n_samples > 0 else 0.0

    # Pre-filter: all events with realized returns
    pre_mask = ~realized_returns.isna()
    n_pre_total = int(pre_mask.sum())

    tx_cost = float(DEFAULT_TRANSACTION_COST)

    if n_pre_total > 0:
        pre_returns = realized_returns[pre_mask]
        raw_label_pre = (pre_returns > tx_cost).astype(int)
        n_pre_pos = int((raw_label_pre == 1).sum())
        n_pre_neg = int((raw_label_pre == 0).sum())
    else:
        pre_returns = pd.Series(dtype=float)
        raw_label_pre = pd.Series(dtype=float)
        n_pre_pos = n_pre_neg = 0

    # Retention metrics
    n_post_total = n_labeled
    n_post_pos = n_positive
    n_post_neg = n_negative

    retention_total = n_post_total / max(n_pre_total, 1)
    retention_pos = n_post_pos / max(n_pre_pos, 1) if n_pre_pos > 0 else 0.0
    retention_neg = n_post_neg / max(n_pre_neg, 1) if n_pre_neg > 0 else 0.0

    def _safe_stats(x: pd.Series) -> Tuple[float, float, int]:
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return 0.0, 0.0, 0
        return float(x_clean.mean()), float(x_clean.std()), len(x_clean)

    # Pre-filter stats (raw economic labels)
    pre_pos_ret = pre_returns[raw_label_pre == 1]
    pre_neg_ret = pre_returns[raw_label_pre == 0]

    pre_pos_mean, pre_pos_std, n_pre_pos_eff = _safe_stats(pre_pos_ret)
    pre_neg_mean, pre_neg_std, n_pre_neg_eff = _safe_stats(pre_neg_ret)

    # Post-filter stats on labeled events
    returns_labeled = realized_returns[labeled_mask]
    labels_clean = binary_labels[labeled_mask]

    post_pos_ret = returns_labeled[labels_clean == 1]
    post_neg_ret = returns_labeled[labels_clean == 0]

    post_pos_mean, post_pos_std, n_post_pos_eff = _safe_stats(post_pos_ret)
    post_neg_mean, post_neg_std, n_post_neg_eff = _safe_stats(post_neg_ret)

    def _cohens_d(m1, s1, n1, m2, s2, n2) -> float:
        if n1 <= 1 or n2 <= 1:
            return float("nan")
        pooled = ((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1)
        if pooled <= 0:
            return float("nan")
        return (m1 - m2) / np.sqrt(pooled)

    d_pre = _cohens_d(pre_pos_mean, pre_pos_std, n_pre_pos_eff, pre_neg_mean, pre_neg_std, n_pre_neg_eff)
    d_post = _cohens_d(post_pos_mean, post_pos_std, n_post_pos_eff, post_neg_mean, post_neg_std, n_post_neg_eff)

    snr_pre = pre_pos_mean / (pre_pos_std + 1e-8) if pre_pos_std > 0 else 0.0
    snr_post = post_pos_mean / (post_pos_std + 1e-8) if post_pos_std > 0 else 0.0

    # Label overlap diagnostic
    overlap_pos_in_neg = int((post_pos_ret < 0).sum())
    overlap_neg_in_pos = int((post_neg_ret > 0).sum())
    total_events_for_overlap = len(post_pos_ret) + len(post_neg_ret)
    if total_events_for_overlap > 0:
        pct_overlap = (overlap_pos_in_neg + overlap_neg_in_pos) / total_events_for_overlap
    else:
        pct_overlap = 0.0

    # Cost-aware event quality
    if len(returns_labeled.dropna()) > 0:
        unconditional_mean = float(returns_labeled.mean())
        frac_small = float((returns_labeled.abs() < tx_cost).mean())
    else:
        unconditional_mean = 0.0
        frac_small = 0.0

    aleatoric_fraction = float(frac_small)
    if aleatoric_fraction < 0.40:
        aleatoric_comment = "Aleatoric fraction < 40%: most error is model/feature-driven; improvement is possible."
    elif aleatoric_fraction < 0.60:
        aleatoric_comment = "Aleatoric fraction 40–60%: mixed noise and model limitations."
    else:
        aleatoric_comment = "Aleatoric fraction > 60%: most unpredictability is intrinsic to the target."

    if len(post_pos_ret.dropna()) > 0:
        mean_pos_ret = float(post_pos_ret.mean())
    else:
        mean_pos_ret = 0.0

    # Prepare isotonic expected returns for bucket diagnostics
    expected_ret = None
    if "target_long" in df.columns or "target_short" in df.columns:
        try:
            if direction == "long" and "target_long" in df.columns:
                expected_ret = df["target_long"].astype(float)
            elif direction == "short" and "target_short" in df.columns:
                expected_ret = df["target_short"].astype(float)
            else:
                # Fallback: combine long/short targets into a single expected return
                tl = df["target_long"].astype(float) if "target_long" in df.columns else pd.Series(0.0, index=df.index)
                ts = df["target_short"].astype(float) if "target_short" in df.columns else pd.Series(0.0, index=df.index)
                expected_ret = tl.where(tl > 0, 0.0) - ts.where(ts > 0, 0.0)
        except Exception:
            expected_ret = None

    # High-probability bucket diagnostics (top-k% by meta_probability),
    # using isotonic expected returns instead of raw realized returns.
    bucket_stats = {}
    prob_series = None
    if "meta_probability" in df.columns and expected_ret is not None:
        prob_series = df["meta_probability"].astype(float)
        valid_bucket_mask = labeled_mask & prob_series.notna() & expected_ret.notna()
        if valid_bucket_mask.any():
            probs_valid = prob_series[valid_bucket_mask]
            rets_valid = expected_ret[valid_bucket_mask]
            labels_valid = binary_labels[valid_bucket_mask]

            bucket_fracs = [0.05, 0.10, 0.20, 0.30, 0.40]
            for frac in bucket_fracs:
                if len(probs_valid) < max(int(1.0 / frac), 50):
                    continue
                try:
                    q = probs_valid.quantile(1.0 - frac)
                    bucket_mask = (probs_valid >= q)
                    if bucket_mask.sum() < 20:
                        continue

                    rets_bucket = rets_valid[bucket_mask]
                    labels_bucket = labels_valid[bucket_mask]
                    win_rate_bucket = float((labels_bucket == 1.0).mean())
                    mean_ret_bucket = float(rets_bucket.mean()) if len(rets_bucket) > 0 else 0.0
                    std_ret_bucket = float(rets_bucket.std()) if len(rets_bucket) > 1 else 0.0
                    sharpe_bucket = mean_ret_bucket / (std_ret_bucket + 1e-8) if std_ret_bucket > 0 else 0.0

                    key = f"top_{int(frac * 100)}"
                    bucket_stats[key] = {
                        "frac": float(frac),
                        "threshold": float(q),
                        "n_events": int(bucket_mask.sum()),
                        "win_rate": float(win_rate_bucket),
                        "mean_expected_return": float(mean_ret_bucket),
                        "sharpe_expected": float(sharpe_bucket),
                    }
                except Exception:
                    continue

    # Volatility-bucket diagnostics (low/mid/high volatility regimes) using
    # volatility_1d when available in labeled_data.
    vol_bucket_stats = {}
    if "volatility_1d" in df.columns:
        try:
            vol = df["volatility_1d"].astype(float)
            vol_mask = labeled_mask & vol.notna()
            if vol_mask.sum() >= 60:
                vol_valid = vol[vol_mask]
                low_thr = float(vol_valid.quantile(1.0 / 3.0))
                high_thr = float(vol_valid.quantile(2.0 / 3.0))

                regimes = {
                    "low": vol < low_thr,
                    "mid": (vol >= low_thr) & (vol < high_thr),
                    "high": vol >= high_thr,
                }

                for name, regime_mask in regimes.items():
                    seg_mask = vol_mask & regime_mask
                    if seg_mask.sum() < 30:
                        continue

                    seg_returns = realized_returns[seg_mask]
                    seg_labels = binary_labels[seg_mask]

                    seg_ret_clean = seg_returns.dropna()
                    seg_labels_clean = seg_labels[~seg_labels.isna()]

                    if len(seg_ret_clean) == 0 or len(seg_labels_clean) == 0:
                        continue

                    seg_mean = float(seg_ret_clean.mean())
                    seg_std = float(seg_ret_clean.std()) if len(seg_ret_clean) > 1 else 0.0
                    seg_sharpe = seg_mean / (seg_std + 1e-8) if seg_std > 0 else 0.0

                    seg_pos = int((seg_labels == 1.0).sum())
                    seg_neg = int((seg_labels == 0.0).sum())
                    seg_total = int(seg_mask.sum())
                    seg_pos_rate = seg_pos / max(seg_pos + seg_neg, 1)

                    vol_bucket_stats[name] = {
                        "n_events": seg_total,
                        "n_positive": seg_pos,
                        "n_negative": seg_neg,
                        "positive_rate": float(seg_pos_rate),
                        "mean_return": float(seg_mean),
                        "sharpe": float(seg_sharpe),
                        "low_threshold": float(low_thr),
                        "high_threshold": float(high_thr),
                    }
        except Exception:
            vol_bucket_stats = {}

    # Simple interpretation helpers for coverage, effect size, SNR and retention
    if coverage < 0.05:
        coverage_comment = "Low coverage (<5%): labels are very sparse; probe models may struggle."
    elif coverage < 0.2:
        coverage_comment = "Moderate coverage (5–20%): typical for event-driven labeling."
    else:
        coverage_comment = "High coverage (>20%): many labeled events; check for redundancy or label noise."

    def _effect_comment(d_val: float) -> str:
        if not np.isfinite(d_val):
            return "Effect size not available (insufficient data)."
        ad = abs(d_val)
        if ad < 0.2:
            return "Very weak separation between positive and negative returns."
        if ad < 0.5:
            return "Small separation between positive and negative returns."
        if ad < 0.8:
            return "Moderate separation between positive and negative returns."
        if ad < 1.5:
            return "Large separation; labels correlate well with economic outcomes."
        return "Very large separation; labels are strongly aligned with economic outcomes."

    effect_post_comment = _effect_comment(d_post)

    if snr_post < 0.5:
        snr_comment = "Low SNR: positive-label returns are noisy relative to their mean."
    elif snr_post < 1.0:
        snr_comment = "Moderate SNR: some signal, but still fairly noisy."
    else:
        snr_comment = "High SNR: positive-label returns are well separated from noise."

    if retention_total < 0.1:
        retention_comment = "Filters are extremely aggressive; only a small fraction of events are kept."
    elif retention_total < 0.3:
        retention_comment = "Filters are moderately aggressive; many events are discarded."
    else:
        retention_comment = "Filters keep a substantial share of events; label density is relatively high."

    noise_ceiling = None
    noise_ceiling_comment = (
        "Noise ceiling requires multiple labelers or repeated labels; "
        "not available in current artifacts."
    )

    def _score_component_lq(value: float, low: float, high: float) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    coverage_score = _score_component_lq(coverage, 0.05, 0.2)
    retention_score = _score_component_lq(retention_total, 0.1, 0.3)
    snr_score = _score_component_lq(snr_post, 0.5, 1.0)
    d_score = _score_component_lq(abs(d_post) if np.isfinite(d_post) else float("nan"), 0.2, 1.5)
    econ_margin = mean_pos_ret - tx_cost
    econ_score = _score_component_lq(econ_margin, 0.0, 0.02)

    label_quality_score_components = [coverage_score, retention_score, snr_score, d_score, econ_score]
    label_quality_score = float(np.mean(label_quality_score_components))

    if label_quality_score < 0.4:
        label_quality_rating = "Bad"
        label_quality_comment = "Low coverage/SNR or weak economic separation; labels are likely noisy or too sparse."
    elif label_quality_score < 0.7:
        label_quality_rating = "Pass"
        label_quality_comment = "Mixed label quality; some usable signal but economic separation or coverage may be modest."
    else:
        label_quality_rating = "Great"
        label_quality_comment = "Strong label quality with good coverage, separation and economic margins."

    # Console output
    print("""
=== Label-Quality Diagnostics ===
""".strip())

    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Total samples: {n_samples}")
    print(f"Labeled samples: {n_labeled} (coverage={coverage:.1%})")
    print(f"Positive labels: {n_positive} ({positive_rate:.1%}), Negative labels: {n_negative}")
    print()

    print("-- Pre vs Post Filter Retention --")
    print(f"Pre-filter events (realized_return not NaN): {n_pre_total}")
    print(f"Pre-filter positive/negative (raw econ > cost): {n_pre_pos} / {n_pre_neg}")
    print(f"Post-filter labeled events: {n_post_total}")
    print(f"Post-filter positive/negative (binary_label): {n_post_pos} / {n_post_neg}")
    print(f"Total retention (post / pre): {retention_total:.1%}")
    print(f"Positive retention: {retention_pos:.1%}")
    print(f"Negative retention: {retention_neg:.1%}")
    print()

    print("-- Economic Separation and SNR --")
    print(f"Pre-filter mean return (label=1/0): {pre_pos_mean:.2%} / {pre_neg_mean:.2%}")
    print(f"Post-filter mean return (label=1/0): {post_pos_mean:.2%} / {post_neg_mean:.2%}")
    print(f"Pre-filter Cohen's d (1 vs 0): {d_pre:.3f}")
    print(f"Post-filter Cohen's d (1 vs 0): {d_post:.3f}")
    print(f"Pre-filter SNR (mean/std, label=1): {snr_pre:.3f}")
    print(f"Post-filter SNR (mean/std, label=1): {snr_post:.3f}")
    print()

    print("-- Label Overlap and Cost-Aware Quality --")
    print(f"Label overlap (mis-signed P&L share): {pct_overlap:.1%}")
    print(f"Transaction cost (approx per event): {tx_cost:.3%}")
    print(f"Unconditional mean event return: {unconditional_mean:.2%}")
    print(f"Mean return (label=1) minus cost: {(mean_pos_ret - tx_cost):.2%}")
    print(f"Fraction of labeled events with |return| < cost: {frac_small:.1%}")

    print()
    print("-- Aleatoric Uncertainty --")
    print(f"Aleatoric uncertainty fraction (|return| < cost): {aleatoric_fraction:.1%}")
    print(f"Interpretation: {aleatoric_comment}")

    if bucket_stats:
        print()
        print("-- High-Probability Buckets (by meta_probability, isotonic expected returns) --")
        for key in sorted(bucket_stats.keys(), key=lambda k: bucket_stats[k]["frac"]):
            stats = bucket_stats[key]
            print(
                f"Top {int(stats['frac']*100):2d}%: n={stats['n_events']}, "
                f"win_rate={stats['win_rate']:.1%}, "
                f"mean_exp_ret={stats['mean_expected_return']:.2%}, "
                f"Sharpe_exp={stats['sharpe_expected']:.2f}"
            )

    if vol_bucket_stats:
        print()
        print("-- Volatility Buckets (by volatility_1d) --")
        for name in ["low", "mid", "high"]:
            if name not in vol_bucket_stats:
                continue
            stats = vol_bucket_stats[name]
            print(
                f"Vol {name:>4}: n={stats['n_events']}, "
                f"pos_rate={stats['positive_rate']:.1%}, "
                f"mean_ret={stats['mean_return']:.2%}, "
                f"Sharpe={stats['sharpe']:.2f}"
            )

    print()
    print("-- Interpretation Hints --")
    print(f"Coverage: {coverage:.1%} → {coverage_comment}")
    print(f"Post-filter effect size (Cohen's d={d_post:.3f}) → {effect_post_comment}")
    print(f"Post-filter SNR (label=1: {snr_post:.3f}) → {snr_comment}")
    print(f"Retention (total={retention_total:.1%}) → {retention_comment}")

    # Export payload
    payload = {
        "section": "label_quality",
        "n_samples": int(n_samples),
        "n_labeled": int(n_labeled),
        "coverage": float(coverage),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "positive_rate": float(positive_rate),
        "pre": {
            "n_total": int(n_pre_total),
            "n_positive": int(n_pre_pos),
            "n_negative": int(n_pre_neg),
            "mean_pos_return": float(pre_pos_mean),
            "mean_neg_return": float(pre_neg_mean),
            "cohens_d": float(d_pre) if np.isfinite(d_pre) else None,
            "snr_pos": float(snr_pre),
        },
        "post": {
            "n_total": int(n_post_total),
            "n_positive": int(n_post_pos),
            "n_negative": int(n_post_neg),
            "mean_pos_return": float(post_pos_mean),
            "mean_neg_return": float(post_neg_mean),
            "cohens_d": float(d_post) if np.isfinite(d_post) else None,
            "snr_pos": float(snr_post),
        },
        "retention": {
            "total": float(retention_total),
            "positive": float(retention_pos),
            "negative": float(retention_neg),
        },
        "overlap": {
            "pct_overlap": float(pct_overlap),
        },
        "cost_metrics": {
            "tx_cost": float(tx_cost),
            "unconditional_mean_return": float(unconditional_mean),
            "mean_pos_minus_cost": float(mean_pos_ret - tx_cost),
            "frac_small_vs_cost": float(frac_small),
        },
        "probability_buckets": bucket_stats,
        "volatility_buckets": vol_bucket_stats,
        "advanced": {
            "aleatoric_uncertainty_fraction": float(aleatoric_fraction),
            "aleatoric_comment": aleatoric_comment,
            "noise_ceiling": noise_ceiling,
            "noise_ceiling_comment": noise_ceiling_comment,
        },
        "summary_score": {
            "score": float(label_quality_score),
            "rating": label_quality_rating,
            "comment": label_quality_comment,
        },
    }

    md_lines = [
        "# SNR Label-Quality Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        "",
        "## Summary",
        f"- Total samples: {n_samples}",
        f"- Labeled samples: {n_labeled} (coverage={coverage:.1%})",
        f"- Positive labels: {n_positive} ({positive_rate:.1%})",
        f"- Negative labels: {n_negative}",
        "",
        "## Retention",
        f"- Pre-filter events (realized_return not NaN): {n_pre_total}",
        f"- Pre-filter pos/neg (raw econ > cost): {n_pre_pos} / {n_pre_neg}",
        f"- Post-filter labeled events: {n_post_total}",
        f"- Post-filter pos/neg (binary_label): {n_post_pos} / {n_post_neg}",
        f"- Total retention: {retention_total:.1%}",
        f"- Positive retention: {retention_pos:.1%}",
        f"- Negative retention: {retention_neg:.1%}",
        "",
        "## Economic Separation and SNR",
        f"- Pre-filter mean return (label=1/0): {pre_pos_mean:.2%} / {pre_neg_mean:.2%}",
        f"- Post-filter mean return (label=1/0): {post_pos_mean:.2%} / {post_neg_mean:.2%}",
        f"- Pre-filter Cohen's d: {d_pre:.3f}",
        f"- Post-filter Cohen's d: {d_post:.3f}",
        f"- Pre-filter SNR (label=1): {snr_pre:.3f}",
        f"- Post-filter SNR (label=1): {snr_post:.3f}",
        "",
        "## Label Overlap and Cost Metrics",
        f"- Label overlap (mis-signed P&L share): {pct_overlap:.1%}",
        f"- Transaction cost (approx per event): {tx_cost:.3%}",
        f"- Unconditional mean event return: {unconditional_mean:.2%}",
        f"- Mean return (label=1) minus cost: {(mean_pos_ret - tx_cost):.2%}",
        f"- Fraction of labeled events with |return| < cost: {frac_small:.1%}",
        f"- Aleatoric uncertainty fraction (|return| < cost): {aleatoric_fraction:.1%}",
        "",
        "## High-Probability Buckets (by meta_probability, isotonic expected returns)",
    ]

    if bucket_stats:
        for key in sorted(bucket_stats.keys(), key=lambda k: bucket_stats[k]["frac"]):
            stats = bucket_stats[key]
            md_lines.append(
                f"- Top {int(stats['frac']*100):2d}%: n={stats['n_events']}, "
                f"win_rate={stats['win_rate']:.1%}, "
                f"mean_exp_ret={stats['mean_expected_return']:.2%}, "
                f"Sharpe_exp={stats['sharpe_expected']:.2f}"
            )
    else:
        md_lines.append("- meta_probability not available or insufficient data for bucket diagnostics.")

    md_lines.extend([
        "",
        "## Volatility Buckets (by volatility_1d)",
    ])

    if vol_bucket_stats:
        for name in ["low", "mid", "high"]:
            if name not in vol_bucket_stats:
                continue
            stats = vol_bucket_stats[name]
            md_lines.append(
                f"- Vol {name}: n={stats['n_events']}, "
                f"pos_rate={stats['positive_rate']:.1%}, "
                f"mean_ret={stats['mean_return']:.2%}, "
                f"Sharpe={stats['sharpe']:.2f}"
            )
    else:
        md_lines.append("- volatility_1d not available or insufficient data for volatility buckets.")

    md_lines.extend([
        "",
        "## Interpretation Hints",
        f"- Coverage ({coverage:.1%}): {coverage_comment}",
        f"- Post-filter effect size (Cohen's d={d_post:.3f}): {effect_post_comment}",
        f"- Post-filter SNR (label=1): {snr_post:.3f} → {snr_comment}",
        f"- Retention (total={retention_total:.1%}): {retention_comment}",
        "",
        "## Overall Label-Quality Score",
        f"- Score (0-1): {label_quality_score:.3f}",
        f"- Rating: {label_quality_rating}",
        f"- Summary: {label_quality_comment}",
    ])

    json_path, md_path = _export_report(
        prefix="snr_label_quality",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# Label-learnability diagnostics
# --------------------------------------------------------------------------------------


def run_label_learnability(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    cv_splits: int = 3,
) -> None:
    """Compute learnability & entropy-based label-quality scores."""
    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)
    X, y = _build_feature_matrix_from_labeled(df)

    learnability, mean_auc = compute_learnability_score(X, y, cv_splits=cv_splits)
    balance = compute_label_entropy_score(y)
    combined, diagnostics = combined_label_quality_objective(
        X,
        y,
        learnability_weight=0.7,
        balance_weight=0.3,
        cv_splits=cv_splits,
    )

    n_valid = int((~y.isna()).sum())
    pos_rate = float(y.mean()) if n_valid > 0 else 0.0

    # Interpretation helpers for learnability and balance
    if mean_auc < 0.55:
        auc_comment = "Mean CV AUC < 0.55 → very weak learnability; labels are close to random."
    elif mean_auc < 0.6:
        auc_comment = "Mean CV AUC 0.55–0.60 → weak but potentially usable signal."
    elif mean_auc < 0.7:
        auc_comment = "Mean CV AUC 0.60–0.70 → moderate learnability."
    else:
        auc_comment = "Mean CV AUC ≥ 0.70 → strong learnability; labels are easy to learn."

    if balance < 0.5:
        balance_comment = "Entropy score < 0.5 → labels are highly imbalanced or dominated by one class."
    elif balance < 0.8:
        balance_comment = "Entropy score 0.5–0.8 → some imbalance but usually acceptable."
    else:
        balance_comment = "Entropy score ≥ 0.8 → labels are well balanced."

    if combined < 0.4:
        combined_comment = "Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds."
    elif combined < 0.6:
        combined_comment = "Combined score 0.4–0.6 → mixed quality; may be adequate for robust models."
    else:
        combined_comment = "Combined score ≥ 0.6 → good overall label quality."

    # Map combined score into [0, 1] summary with rating
    learnability_score = float(max(0.0, min(1.0, combined)))
    if learnability_score < 0.4:
        learnability_rating = "Bad"
    elif learnability_score < 0.6:
        learnability_rating = "Pass"
    else:
        learnability_rating = "Great"

    # Console output
    print("""
=== Label-Learnability Diagnostics ===
""".strip())
    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Valid labeled samples: {n_valid}")
    print(f"Positive label rate: {pos_rate:.1%}")
    print()

    print("-- Learnability (Probe Model AUC) --")
    print(f"Mean CV AUC: {mean_auc:.4f}")
    print(f"Learnability score (AUC - 0.5 * std): {learnability:.4f}")
    print()

    print("-- Entropy / Balance --")
    print(f"Entropy-based balance score: {balance:.4f}")
    print()

    print("-- Combined Label-Quality Objective (0.7 * learnability + 0.3 * balance) --")
    print(f"Combined score: {combined:.4f}")
    print()

    print("Diagnostics snapshot:")
    for k in sorted(diagnostics.keys()):
        v = diagnostics[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print()
    print("-- Interpretation Hints --")
    print(f"Learnability (mean AUC={mean_auc:.4f}) → {auc_comment}")
    print(f"Balance (entropy score={balance:.4f}) → {balance_comment}")
    print(f"Combined score ({combined:.4f}) → {combined_comment}")

    # Export payload
    payload = {
        "section": "label_learnability",
        "n_valid": int(n_valid),
        "positive_rate": float(pos_rate),
        "learnability": float(learnability),
        "mean_auc": float(mean_auc),
        "balance": float(balance),
        "combined": float(combined),
        "diagnostics": diagnostics,
        "summary_score": {
            "score": learnability_score,
            "rating": learnability_rating,
            "comment": combined_comment,
        },
    }

    md_lines = [
        "# Label-Learnability Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        "",
        "## Summary",
        f"- Valid labeled samples: {n_valid}",
        f"- Positive label rate: {pos_rate:.1%}",
        "",
        "## Learnability",
        f"- Mean CV AUC: {mean_auc:.4f}",
        f"- Learnability score (AUC - 0.5 * std): {learnability:.4f}",
        "",
        "## Entropy / Balance",
        f"- Balance score: {balance:.4f}",
        "",
        "## Combined Label-Quality Objective",
        f"- Combined score: {combined:.4f}",
        "",
        "## Interpretation Hints",
        f"- Learnability (mean AUC={mean_auc:.4f}): {auc_comment}",
        f"- Balance (entropy score={balance:.4f}): {balance_comment}",
        f"- Combined score ({combined:.4f}): {combined_comment}",
        "",
        "## Overall Learnability Score",
        f"- Score (0-1): {learnability_score:.3f}",
        f"- Rating: {learnability_rating}",
        f"- Summary: {combined_comment}",
    ]

    json_path, md_path = _export_report(
        prefix="snr_label_learnability",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# Model-robustness diagnostics
# --------------------------------------------------------------------------------------


def run_model_robustness(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    cv_splits: int = 5,
) -> None:
    """Run a probe LightGBM model with time-series CV to assess robustness.

    Reports per-fold AUC, Brier score, PR-AUC, and summary statistics.
    """
    df = _load_labeled_data(symbol, exchange, timeframe, direction=direction, model=model)
    X, y = _build_feature_matrix_from_labeled(df)

    y_array = y.values.astype(float)
    X_array = X.values.astype(float)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    fold_metrics = []
    all_y_true = []
    all_p_pred = []
    all_p_baseline = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_array), start=1):
        X_tr, X_te = X_array[tr_idx], X_array[te_idx]
        y_tr, y_te = y_array[tr_idx], y_array[te_idx]

        # Require both classes in train and test for meaningful AUC
        if len(np.unique(y_tr[~np.isnan(y_tr)])) < 2 or len(np.unique(y_te[~np.isnan(y_te)])) < 2:
            continue

        # Clean NaNs in labels consistently between X and y
        mask_tr = ~np.isnan(y_tr)
        y_tr_clean = y_tr[mask_tr]
        X_tr_clean = X_tr[mask_tr]
        mask_te = ~np.isnan(y_te)
        y_te_clean = y_te[mask_te]
        X_te_clean = X_te[mask_te]

        if len(y_tr_clean) < 50 or len(y_te_clean) < 20:
            continue

        clf = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=50,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        )

        clf.fit(X_tr_clean, y_tr_clean)
        prob = clf.predict_proba(X_te_clean)[:, 1]

        # Logistic regression model for model-family comparison
        log_clf = LogisticRegression(solver="lbfgs", max_iter=200)
        log_clf.fit(X_tr_clean, y_tr_clean)
        log_prob = log_clf.predict_proba(X_te_clean)[:, 1]

        # Naive baseline: constant probability equal to training positive rate
        pos_rate_tr = float(np.nanmean(y_tr_clean)) if len(y_tr_clean) > 0 else 0.5
        baseline_prob = np.full_like(prob, fill_value=pos_rate_tr, dtype=float)

        all_y_true.append(y_te_clean)
        all_p_pred.append(prob)
        all_p_baseline.append(baseline_prob)

        try:
            auc = roc_auc_score(y_te_clean, prob)
        except Exception:
            auc = float("nan")

        try:
            brier = brier_score_loss(y_te_clean, prob)
        except Exception:
            brier = float("nan")

        try:
            ap = average_precision_score(y_te_clean, prob)
        except Exception:
            ap = float("nan")

        # Logistic regression metrics
        try:
            auc_log = roc_auc_score(y_te_clean, log_prob)
        except Exception:
            auc_log = float("nan")

        try:
            brier_log = brier_score_loss(y_te_clean, log_prob)
        except Exception:
            brier_log = float("nan")

        try:
            ap_log = average_precision_score(y_te_clean, log_prob)
        except Exception:
            ap_log = float("nan")

        fold_metrics.append({
            "fold": fold_idx,
            "n_train": int(len(y_tr_clean)),
            "n_test": int(len(y_te_clean)),
            "auc": float(auc) if np.isfinite(auc) else float("nan"),
            "brier": float(brier) if np.isfinite(brier) else float("nan"),
            "ap": float(ap) if np.isfinite(ap) else float("nan"),
            "auc_logistic": float(auc_log) if np.isfinite(auc_log) else float("nan"),
            "brier_logistic": float(brier_log) if np.isfinite(brier_log) else float("nan"),
            "ap_logistic": float(ap_log) if np.isfinite(ap_log) else float("nan"),
        })

    if not fold_metrics:
        print("No valid CV folds for robustness diagnostics (insufficient data or degenerate labels).")
        return

    aucs = np.array([m["auc"] for m in fold_metrics], dtype=float)
    briers = np.array([m["brier"] for m in fold_metrics], dtype=float)
    aps = np.array([m["ap"] for m in fold_metrics], dtype=float)

    # Logistic regression metrics per fold
    aucs_log = np.array([m.get("auc_logistic", float("nan")) for m in fold_metrics], dtype=float)
    briers_log = np.array([m.get("brier_logistic", float("nan")) for m in fold_metrics], dtype=float)
    aps_log = np.array([m.get("ap_logistic", float("nan")) for m in fold_metrics], dtype=float)

    mean_auc = float(np.nanmean(aucs))
    std_auc = float(np.nanstd(aucs))

    mean_brier = float(np.nanmean(briers))
    std_brier = float(np.nanstd(briers))

    mean_ap = float(np.nanmean(aps))
    std_ap = float(np.nanstd(aps))

    mean_auc_log = float(np.nanmean(aucs_log)) if np.isfinite(aucs_log).any() else float("nan")
    mean_brier_log = float(np.nanmean(briers_log)) if np.isfinite(briers_log).any() else float("nan")
    mean_ap_log = float(np.nanmean(aps_log)) if np.isfinite(aps_log).any() else float("nan")

    stability_score = 1.0 - (std_auc / (mean_auc + 1e-9)) if np.isfinite(mean_auc) else 0.0

    # Aggregate predictions across folds for advanced diagnostics
    if all_y_true:
        y_all = np.concatenate(all_y_true)
        p_all = np.concatenate(all_p_pred)
        p_base_all = np.concatenate(all_p_baseline)
    else:
        y_all = np.array([])
        p_all = np.array([])
        p_base_all = np.array([])

    pseudo_r2 = float("nan")
    model_snr = float("nan")
    auc_global = float("nan")
    perm_pvalue = float("nan")
    baseline_auc = float("nan")
    baseline_brier = float("nan")
    baseline_ap = float("nan")
    delta_auc = float("nan")
    delta_brier = float("nan")
    delta_ap = float("nan")
    pseudo_r2_ci_low = float("nan")
    pseudo_r2_ci_high = float("nan")
    residual_pattern_strength = float("nan")
    residual_lag1_autocorr = float("nan")

    if y_all.size > 0:
        # Pseudo-R^2 on probabilities: 1 - SSE/SST
        try:
            y_mean = float(np.mean(y_all))
            sse = float(np.sum((y_all - p_all) ** 2))
            sst = float(np.sum((y_all - y_mean) ** 2))
            if sst > 0:
                pseudo_r2 = 1.0 - sse / sst
        except Exception:
            pseudo_r2 = float("nan")

        # Residual diagnostics (pattern strength and autocorrelation)
        try:
            residuals = y_all - p_all
            if residuals.size > 1:
                # Pattern strength: max - min mean residual across probability deciles
                try:
                    quantiles = np.quantile(p_all, np.linspace(0.0, 1.0, 11))
                    bucket_means: list[float] = []
                    for i in range(10):
                        lo, hi = quantiles[i], quantiles[i + 1]
                        mask = (p_all >= lo) & (p_all <= hi)
                        if np.any(mask):
                            bucket_means.append(float(np.mean(residuals[mask])))
                    if bucket_means:
                        residual_pattern_strength = float(max(bucket_means) - min(bucket_means))
                except Exception:
                    residual_pattern_strength = float("nan")

                # Lag-1 autocorrelation of residuals (time-ordered across folds)
                try:
                    r0 = residuals[:-1]
                    r1 = residuals[1:]
                    if r0.size > 1 and np.std(r0) > 0 and np.std(r1) > 0:
                        corr_matrix = np.corrcoef(r0, r1)
                        residual_lag1_autocorr = float(corr_matrix[0, 1])
                except Exception:
                    residual_lag1_autocorr = float("nan")
        except Exception:
            residual_pattern_strength = float("nan")
            residual_lag1_autocorr = float("nan")

        # Model-level SNR: separation of predicted probabilities for pos vs neg labels
        try:
            pos_mask = y_all == 1.0
            neg_mask = y_all == 0.0
            p_pos = p_all[pos_mask]
            p_neg = p_all[neg_mask]
            if len(p_pos) > 1 and len(p_neg) > 1:
                mean_pos = float(np.mean(p_pos))
                mean_neg = float(np.mean(p_neg))
                std_pos = float(np.std(p_pos))
                std_neg = float(np.std(p_neg))
                denom = len(p_pos) + len(p_neg) - 2
                if denom > 0:
                    pooled_var = ((len(p_pos) - 1) * std_pos ** 2 + (len(p_neg) - 1) * std_neg ** 2) / denom
                    pooled_std = float(np.sqrt(pooled_var))
                    if pooled_std > 0 and np.isfinite(pooled_std):
                        model_snr = (mean_pos - mean_neg) / pooled_std
        except Exception:
            model_snr = float("nan")

        # Global AUC across all folds
        try:
            auc_global = float(roc_auc_score(y_all, p_all))
        except Exception:
            auc_global = float("nan")

        # Permutation p-value for global AUC
        if np.isfinite(auc_global) and y_all.size >= 100:
            rng = np.random.default_rng(42)
            perm_aucs: list[float] = []
            for _ in range(200):
                y_perm = rng.permutation(y_all)
                try:
                    perm_auc = roc_auc_score(y_perm, p_all)
                except Exception:
                    continue
                if np.isfinite(perm_auc):
                    perm_aucs.append(float(perm_auc))
            if perm_aucs:
                perm_arr = np.array(perm_aucs, dtype=float)
                perm_pvalue = float((np.sum(perm_arr >= auc_global) + 1) / (len(perm_arr) + 1))

        # Baseline metrics (constant probability) aggregated across folds
        try:
            baseline_auc = float(roc_auc_score(y_all, p_base_all))
        except Exception:
            baseline_auc = float("nan")
        try:
            baseline_brier = float(brier_score_loss(y_all, p_base_all))
        except Exception:
            baseline_brier = float("nan")
        try:
            baseline_ap = float(average_precision_score(y_all, p_base_all))
        except Exception:
            baseline_ap = float("nan")

        if np.isfinite(baseline_auc) and np.isfinite(mean_auc):
            delta_auc = mean_auc - baseline_auc
        if np.isfinite(baseline_brier) and np.isfinite(mean_brier):
            # Lower Brier is better; positive delta means improvement vs baseline
            delta_brier = baseline_brier - mean_brier
        if np.isfinite(baseline_ap) and np.isfinite(mean_ap):
            delta_ap = mean_ap - baseline_ap

        # Bootstrap CI for pseudo-R^2
        if np.isfinite(pseudo_r2) and y_all.size >= 100:
            rng_ci = np.random.default_rng(123)
            boot_stats: list[float] = []
            for _ in range(200):
                idx = rng_ci.integers(0, y_all.size, size=y_all.size)
                y_boot = y_all[idx]
                p_boot = p_all[idx]
                try:
                    y_mean_boot = float(np.mean(y_boot))
                    sse_boot = float(np.sum((y_boot - p_boot) ** 2))
                    sst_boot = float(np.sum((y_boot - y_mean_boot) ** 2))
                    if sst_boot > 0:
                        boot_r2 = 1.0 - sse_boot / sst_boot
                        if np.isfinite(boot_r2):
                            boot_stats.append(float(boot_r2))
                except Exception:
                    continue
            if boot_stats:
                boot_arr = np.array(boot_stats, dtype=float)
                pseudo_r2_ci_low = float(np.percentile(boot_arr, 2.5))
                pseudo_r2_ci_high = float(np.percentile(boot_arr, 97.5))

    # Model family comparison comment (LightGBM vs LogisticRegression)
    model_family_comment = "N/A"
    if np.isfinite(mean_auc) and np.isfinite(mean_auc_log):
        diff = mean_auc - mean_auc_log
        if diff > 0.02:
            model_family_comment = "Nonlinear (LightGBM) >> linear (Logistic); real nonlinear structure present."
        elif diff < -0.02:
            model_family_comment = "Linear >> nonlinear; tree model may be overfitting or mis-specified."
        else:
            if mean_auc >= 0.6 and mean_auc_log >= 0.6:
                model_family_comment = "All models perform similarly well; problem is stable and well-posed."
            else:
                model_family_comment = "All models perform similarly poorly; target has low intrinsic predictability."

    # Interpretation helpers for robustness
    if mean_auc < 0.55:
        auc_comment = "Mean CV AUC < 0.55 → robust models may still struggle; signal is weak."
    elif mean_auc < 0.6:
        auc_comment = "Mean CV AUC 0.55–0.60 → weak but potentially exploitable signal."
    elif mean_auc < 0.7:
        auc_comment = "Mean CV AUC 0.60–0.70 → moderate predictive power."
    else:
        auc_comment = "Mean CV AUC ≥ 0.70 → strong predictive power for the probe model."

    if stability_score < 0.8:
        stability_comment = "Stability score < 0.8 → performance is quite unstable across time splits."
    elif stability_score < 0.9:
        stability_comment = "Stability score 0.8–0.9 → moderate stability; some variation across folds."
    else:
        stability_comment = "Stability score ≥ 0.9 → highly stable performance across folds."

    if mean_brier > 0.25:
        brier_comment = "Mean Brier > 0.25 → probabilities are poorly calibrated or close to random."
    elif mean_brier > 0.18:
        brier_comment = "Mean Brier 0.18–0.25 → moderate calibration; room for improvement."
    else:
        brier_comment = "Mean Brier ≤ 0.18 → reasonably well-calibrated probabilities."

    # Map robustness into [0, 1] summary score with rating
    def _score_component_mr(value: float, low: float, high: float, invert: bool = False) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        if invert:
            # Lower is better
            if value >= high:
                return 0.0
            if value <= low:
                return 1.0
            return float((high - value) / (high - low))
        # Higher is better
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return float((value - low) / (high - low))

    auc_score = _score_component_mr(mean_auc, 0.55, 0.70)
    stability_score_norm = _score_component_mr(stability_score, 0.80, 0.90)
    brier_score_norm = _score_component_mr(mean_brier, 0.18, 0.25, invert=True)

    robustness_score_components = [auc_score, stability_score_norm, brier_score_norm]
    robustness_score = float(np.mean(robustness_score_components))

    if robustness_score < 0.4:
        robustness_rating = "Bad"
        robustness_comment = "Probe model is weak or unstable across folds."
    elif robustness_score < 0.7:
        robustness_rating = "Pass"
        robustness_comment = "Moderate robustness; some time variation or calibration issues."
    else:
        robustness_rating = "Great"
        robustness_comment = "Strong, stable probe model with consistent performance."

    def _fmt(value, digits: int = 4) -> str:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"

    # Console output
    print("""
=== Model-Robustness Diagnostics (Probe LightGBM) ===
""".strip())
    print(f"Symbol: {symbol} | Exchange: {exchange} | Timeframe: {timeframe}")
    print(f"Folds evaluated: {len(fold_metrics)} (requested: {cv_splits})")
    print()

    print("Per-fold metrics:")
    for m in fold_metrics:
        print(
            f"  Fold {m['fold']}: n_train={m['n_train']}, n_test={m['n_test']}, "
            f"AUC={m['auc']:.4f}, Brier={m['brier']:.4f}, AP={m['ap']:.4f}"
        )

    print()
    print("Summary:")
    print(f"  Mean AUC: {mean_auc:.4f} (std={std_auc:.4f})")
    print(f"  Mean Brier: {mean_brier:.4f} (std={std_brier:.4f})")
    print(f"  Mean AP: {mean_ap:.4f} (std={std_ap:.4f})")
    print(f"  Stability score (1 - std(AUC)/mean(AUC)): {stability_score:.4f}")

    print()
    print("-- Interpretation Hints --")
    print(f"Mean AUC ({mean_auc:.4f}) → {auc_comment}")
    print(f"Stability score ({stability_score:.4f}) → {stability_comment}")
    print(f"Mean Brier ({mean_brier:.4f}) → {brier_comment}")

    print()
    print("-- Advanced Robustness Diagnostics --")
    print(f"Pseudo-R^2 (y vs predicted prob): {_fmt(pseudo_r2)}")
    print(
        f"Pseudo-R^2 95% CI: "
        f"[{_fmt(pseudo_r2_ci_low)}, {_fmt(pseudo_r2_ci_high)}]"
    )
    print(f"Global AUC (all folds combined): {_fmt(auc_global)}")
    print(f"Permutation p-value for global AUC: {_fmt(perm_pvalue)}")
    print(f"Model-level SNR (p_hat pos vs neg): {_fmt(model_snr)}")

    print()
    print("-- Naive Baseline Comparison (constant probability) --")
    print(f"Baseline AUC: {_fmt(baseline_auc)} | Probe AUC: {_fmt(mean_auc)} | Delta: {_fmt(delta_auc)}")
    print(
        f"Baseline Brier: {_fmt(baseline_brier)} | Probe Brier: {_fmt(mean_brier)} "
        f"| Delta (baseline - probe): {_fmt(delta_brier)}"
    )
    print(f"Baseline AP: {_fmt(baseline_ap)} | Probe AP: {_fmt(mean_ap)} | Delta: {_fmt(delta_ap)}")

    print()
    print("-- Residual Diagnostics --")
    print(
        "Residual pattern strength (max - min mean residual across probability deciles): "
        f"{_fmt(residual_pattern_strength)}"
    )
    print(f"Residual lag-1 autocorrelation: {_fmt(residual_lag1_autocorr)}")

    print()
    print("-- Model Family Comparison (LightGBM vs LogisticRegression) --")
    print(
        f"Mean AUC LightGBM: {_fmt(mean_auc)} | "
        f"LogisticRegression: {_fmt(mean_auc_log)}"
    )
    print(f"Model-family comment: {model_family_comment}")

    # Export payload
    payload = {
        "section": "model_robustness",
        "cv_splits": int(cv_splits),
        "fold_metrics": fold_metrics,
        "summary": {
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc),
            "mean_brier": float(mean_brier),
            "std_brier": float(std_brier),
            "mean_ap": float(mean_ap),
            "std_ap": float(std_ap),
            "stability_score": float(stability_score),
            "n_folds": int(len(fold_metrics)),
        },
        "advanced": {
            "global_auc": float(auc_global) if np.isfinite(auc_global) else None,
            "pseudo_r2": float(pseudo_r2) if np.isfinite(pseudo_r2) else None,
            "pseudo_r2_ci_low": float(pseudo_r2_ci_low) if np.isfinite(pseudo_r2_ci_low) else None,
            "pseudo_r2_ci_high": float(pseudo_r2_ci_high) if np.isfinite(pseudo_r2_ci_high) else None,
            "model_snr": float(model_snr) if np.isfinite(model_snr) else None,
            "perm_pvalue_auc": float(perm_pvalue) if np.isfinite(perm_pvalue) else None,
            "residual_pattern_strength": float(residual_pattern_strength)
            if np.isfinite(residual_pattern_strength)
            else None,
            "residual_lag1_autocorr": float(residual_lag1_autocorr)
            if np.isfinite(residual_lag1_autocorr)
            else None,
            "model_family": {
                "mean_auc_lightgbm": float(mean_auc) if np.isfinite(mean_auc) else None,
                "mean_auc_logistic": float(mean_auc_log) if np.isfinite(mean_auc_log) else None,
                "comment": model_family_comment,
            },
            "baseline": {
                "auc": float(baseline_auc) if np.isfinite(baseline_auc) else None,
                "brier": float(baseline_brier) if np.isfinite(baseline_brier) else None,
                "ap": float(baseline_ap) if np.isfinite(baseline_ap) else None,
                "delta_auc": float(delta_auc) if np.isfinite(delta_auc) else None,
                "delta_brier": float(delta_brier) if np.isfinite(delta_brier) else None,
                "delta_ap": float(delta_ap) if np.isfinite(delta_ap) else None,
            },
        },
        "summary_score": {
            "score": float(robustness_score),
            "rating": robustness_rating,
            "comment": robustness_comment,
        },
    }

    md_lines = [
        "# Model-Robustness Diagnostics (Probe LightGBM)",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        "",
        "## Fold Metrics",
    ]

    for m in fold_metrics:
        md_lines.append(
            f"- Fold {m['fold']}: n_train={m['n_train']}, n_test={m['n_test']}, "
            f"AUC={m['auc']:.4f}, Brier={m['brier']:.4f}, AP={m['ap']:.4f}"
        )

    md_lines.extend(
        [
            "",
            "## Summary",
            f"- Mean AUC: {mean_auc:.4f} (std={std_auc:.4f})",
            f"- Mean Brier: {mean_brier:.4f} (std={std_brier:.4f})",
            f"- Mean AP: {mean_ap:.4f} (std={std_ap:.4f})",
            f"- Stability score (1 - std(AUC)/mean(AUC)): {stability_score:.4f}",
            "",
            "## Interpretation Hints",
            f"- Mean AUC ({mean_auc:.4f}): {auc_comment}",
            f"- Stability score ({stability_score:.4f}): {stability_comment}",
            f"- Mean Brier ({mean_brier:.4f}): {brier_comment}",
            "",
            "## Advanced Robustness Diagnostics",
            f"- Global AUC (all folds combined): {_fmt(auc_global)}",
            f"- Pseudo-R^2 (y vs predicted prob): {_fmt(pseudo_r2)}",
            f"- Pseudo-R^2 95% CI: [{_fmt(pseudo_r2_ci_low)}, {_fmt(pseudo_r2_ci_high)}]",
            f"- Permutation p-value for global AUC: {_fmt(perm_pvalue)}",
            f"- Model-level SNR (p_hat pos vs neg): {_fmt(model_snr)}",
            "",
            "## Naive Baseline Comparison (constant probability)",
            f"- Baseline AUC: {_fmt(baseline_auc)} | Probe AUC: {_fmt(mean_auc)} | Delta: {_fmt(delta_auc)}",
            f"- Baseline Brier: {_fmt(baseline_brier)} | Probe Brier: {_fmt(mean_brier)} | Delta (baseline - probe): {_fmt(delta_brier)}",
            f"- Baseline AP: {_fmt(baseline_ap)} | Probe AP: {_fmt(mean_ap)} | Delta: {_fmt(delta_ap)}",
            "",
            "## Residual Diagnostics",
            "- Residual pattern strength (max - min mean residual across probability deciles): "
            f"{_fmt(residual_pattern_strength)}",
            f"- Residual lag-1 autocorrelation: {_fmt(residual_lag1_autocorr)}",
            "",
            "## Model Family Comparison (LightGBM vs LogisticRegression)",
            f"- Mean AUC LightGBM: {_fmt(mean_auc)} | LogisticRegression: {_fmt(mean_auc_log)}",
            f"- Comment: {model_family_comment}",
            "",
            "## Overall Model-Robustness Score",
            f"- Score (0-1): {robustness_score:.3f}",
            f"- Rating: {robustness_rating}",
            f"- Summary: {robustness_comment}",
        ]
    )

    json_path, md_path = _export_report(
        prefix="snr_model_robustness",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )

    print(f"\nReports saved to: {json_path} and {md_path}")


def run_full(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    model: str = "analyst",
    cv_splits_learn: int = 3,
    cv_splits_robust: int = 5,
) -> None:
    _LAST_EXPORTS.clear()

    run_label_quality(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
    )

    run_label_learnability(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        cv_splits=cv_splits_learn,
    )

    run_model_robustness(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        cv_splits=cv_splits_robust,
    )

    required_prefixes = [
        "snr_label_quality",
        "snr_label_learnability",
        "snr_model_robustness",
    ]
    missing = [p for p in required_prefixes if p not in _LAST_EXPORTS]
    if missing:
        logger.warning("Missing diagnostics for prefixes: %s", ", ".join(missing))
        return

    lq = _LAST_EXPORTS["snr_label_quality"]
    ll = _LAST_EXPORTS["snr_label_learnability"]
    mr = _LAST_EXPORTS["snr_model_robustness"]

    lq_payload = lq["payload"]
    ll_payload = ll["payload"]
    mr_payload = mr["payload"]

    lq_coverage = lq_payload.get("coverage")
    lq_positive_rate = lq_payload.get("positive_rate")
    lq_post = lq_payload.get("post", {})
    lq_snr_post = lq_post.get("snr_pos")
    lq_cohens_d = lq_post.get("cohens_d")

    learnability = ll_payload.get("learnability")
    learn_mean_auc = ll_payload.get("mean_auc")
    balance = ll_payload.get("balance")
    combined = ll_payload.get("combined")

    mr_summary = mr_payload.get("summary", {})
    mr_mean_auc = mr_summary.get("mean_auc")
    mr_stability = mr_summary.get("stability_score")
    mr_mean_brier = mr_summary.get("mean_brier")

    mr_advanced = mr_payload.get("advanced", {}) if isinstance(mr_payload, dict) else {}
    mr_global_auc = mr_advanced.get("global_auc") if isinstance(mr_advanced, dict) else None
    mr_pseudo_r2 = mr_advanced.get("pseudo_r2") if isinstance(mr_advanced, dict) else None
    mr_perm_p = mr_advanced.get("perm_pvalue_auc") if isinstance(mr_advanced, dict) else None
    mr_baseline = mr_advanced.get("baseline", {}) if isinstance(mr_advanced, dict) else {}
    mr_delta_auc = mr_baseline.get("delta_auc") if isinstance(mr_baseline, dict) else None
    mr_delta_brier = mr_baseline.get("delta_brier") if isinstance(mr_baseline, dict) else None
    mr_delta_ap = mr_baseline.get("delta_ap") if isinstance(mr_baseline, dict) else None

    lq_summary_score = lq_payload.get("summary_score", {})
    ll_summary_score = ll_payload.get("summary_score", {})
    mr_summary_score = mr_payload.get("summary_score", {})

    lq_score = lq_summary_score.get("score")
    lq_rating = lq_summary_score.get("rating")
    ll_score = ll_summary_score.get("score")
    ll_rating = ll_summary_score.get("rating")
    mr_score = mr_summary_score.get("score")
    mr_rating = mr_summary_score.get("rating")

    lq_advanced = lq_payload.get("advanced", {})
    aleatoric_fraction = lq_advanced.get("aleatoric_uncertainty_fraction")

    def _fmt_pct(value) -> str:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.1%}"

    def _fmt_float(value, digits: int = 4) -> str:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"

    md_lines: list[str] = [
        "# Full SNR Diagnostics Report",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        f"**Direction**: {direction}",
        f"**Model**: {model}",
        "",
        "## High-Level Summary",
        f"- Label coverage: {_fmt_pct(lq_coverage)} (labeled / total samples)",
        f"- Label positive rate: {_fmt_pct(lq_positive_rate)}",
        f"- Label economic SNR (post-filter, label=1): {_fmt_float(lq_snr_post, digits=3)}",
        f"- Label effect size (post-filter Cohen's d): {_fmt_float(lq_cohens_d, digits=3)}",
        f"- Aleatoric uncertainty fraction (|return| < cost): {_fmt_pct(aleatoric_fraction)}",
        "",
        f"- Learnability mean CV AUC: {_fmt_float(learn_mean_auc, digits=4)}",
        f"- Learnability score (AUC - 0.5 * std): {_fmt_float(learnability, digits=4)}",
        f"- Label balance (entropy score): {_fmt_float(balance, digits=4)}",
        f"- Combined label-quality score: {_fmt_float(combined, digits=4)}",
        "",
        f"- Probe model mean AUC: {_fmt_float(mr_mean_auc, digits=4)}",
        f"- Probe model stability score: {_fmt_float(mr_stability, digits=4)}",
        f"- Probe model mean Brier score: {_fmt_float(mr_mean_brier, digits=4)}",
        f"- Probe global AUC (all folds combined): {_fmt_float(mr_global_auc, digits=4)}",
        f"- Probe pseudo-R^2 (y vs predicted prob): {_fmt_float(mr_pseudo_r2, digits=4)}",
        f"- Probe permutation p-value (AUC): {_fmt_float(mr_perm_p, digits=3)}",
        f"- Probe vs baseline ΔAUC: {_fmt_float(mr_delta_auc, digits=4)}, ΔBrier (baseline - probe): {_fmt_float(mr_delta_brier, digits=4)}, ΔAP: {_fmt_float(mr_delta_ap, digits=4)}",
        "",
        f"- Label-quality summary score: {_fmt_float(lq_score, digits=3)} (Rating: {lq_rating or 'N/A'})",
        f"- Learnability summary score: {_fmt_float(ll_score, digits=3)} (Rating: {ll_rating or 'N/A'})",
        f"- Model-robustness summary score: {_fmt_float(mr_score, digits=3)} (Rating: {mr_rating or 'N/A'})",
        "",
        "## Metric Definitions (brief)",
        "- **Coverage**: share of events that receive a binary label.",
        "- **Positive rate**: fraction of labeled events with label=1.",
        "- **Cohen's d**: standardized difference in mean returns between positive and negative labels.",
        "- **SNR (mean/std)**: mean positive-label return divided by its standard deviation.",
        "- **Learnability AUC**: mean cross-validated ROC AUC from a shallow probe model.",
        "- **Learnability score**: AUC penalized by instability (AUC - 0.5 * std).",
        "- **Entropy balance**: how balanced labels are between 0 and 1; 1.0 is 50/50.",
        "- **Combined score**: weighted average of learnability and balance.",
        "- **Brier score**: mean squared error between predicted probabilities and true labels; lower is better.",
        "- **Stability score**: 1 - std(AUC)/mean(AUC); higher indicates more stable performance across folds.",
        "",
        "## Detailed Diagnostics",
        "",
        "### Label-Quality",
    ]

    lq_md = lq["markdown_lines"]
    ll_md = ll["markdown_lines"]
    mr_md = mr["markdown_lines"]

    md_lines.extend(lq_md[2:] if len(lq_md) > 2 else lq_md)
    md_lines.extend([
        "",
        "### Label-Learnability",
    ])
    md_lines.extend(ll_md[2:] if len(ll_md) > 2 else ll_md)
    md_lines.extend([
        "",
        "### Model-Robustness",
    ])
    md_lines.extend(mr_md[2:] if len(mr_md) > 2 else mr_md)

    md_lines.extend([
        "",
        "## Label Quality, Learnability and Robustness Reference",
        "",
        "### Label quality",
        "1. Noise Ceiling (if multiple labelers / repeated labels). If you have multiple labelers, this can be combined with inter-rater reliability metrics (ICC, Cohen09s kappa).",
        "> 0.6 b Labels are internally consistent; high R00 is achievable.",
        "0.40.6 b Labels moderately noisy; realistic ceilings are limited.",
        "< 0.4 b Labels are extremely noisy; even perfect models cannot perform well.",
        "",
        "2. Aleatoric Uncertainty Fraction. Could link it to expected max R00; i.e., intrinsic unpredictability sets a ceiling for achievable performance",
        "< 40% b Most error is model/feature-driven; improvement is possible.",
        "4060% b Mixed noise and model limitations.",
        "> 60% b Most unpredictability is intrinsic to the target.",
        "",
        "### Label learnability vs noise",
        "1. R00. Low R00 could be due to missing features or poor model choice, not just label noise",
        "R00 > 0.40 b The target has a strong predictable signal; meaningful modeling gains are possible.",
        "0.10 < R00 0.40 b The target has a weakbmoderate signal; features matter more than model choice.",
        "R00 0.10 b The target is barely predictable; noise likely dominates.",
        "",
        "2. SNR",
        "SNR > 1 b Signal is stronger than noise; the target is learnable.",
        "0.3 < SNR 1 b Weak but real signal exists; more features or nonlinear models may help.",
        "SNR 0.3 b Noise overwhelms signal; predictability is fundamentally low.",
        "",
        "3. Permutation p-value. If p is high, it may indicate noisy labels, but it could also reflect poor features or an underpowered model.",
        "p < 0.01 b The model captures a real, statistically robust pattern.",
        "0.01 c p 0.20 b There might be signal, but itb s weak or unstable.",
        "p > 0.20 b The model performs no better than chance; label likely noisy.",
        "",
        "4. Naive Baselines. A very simple predictive model used as a reference point. Establishes a floor for model performance & distinguish real signal from noise:",
        "Model 4 baseline b low predictability, focus on labels or features",
        "Model >> baseline b real signal exists, worth improving features/model (doesn't say we haven't reached the ceiling)",
        "",
        "### Model & features robustness",
        "1. Bootstrap R00 Confidence Interval. Helps assess stability and reliability of model performance, helps detect overfitting if the CI is very wide or unstable across bootstraps",
        "CI does NOT include 0 b Performance is reliably above noise level.",
        "CI barely clears 0 (lower bound < 0.05) b Signal is present but fragile.",
        "CI spans below 0 b Model performance might be indistinguishable from noise.",
        "",
        "2. Residual Structure. Residual structure tells you what signal your model/features are missing (and if there is a pattern), not directly about label noise.",
        "Residuals look random b The model extracted essentially all available signal.",
        "Residuals show patterns b There is remaining structure the model/features are missing.",
        "Residuals differ strongly across subgroups b Predictability varies by segment (not globally noisy).",
        "",
        "3. Residual Autocorrelation. Measures whether residuals are temporally or sequentially correlated (often lag-1 autocorrelation). Even if R00 looks okay, autocorrelated residuals indicate hidden structure your features/model missed.",
        "Lag-1 autocorr < 0.10 b No missing temporal/ordered structure.",
        "0.100.20 b Some time dependence is not modeled.",
        "> 0.20 b Strong sequential structure missing; target not fully explained.",
        "",
        "4. Model Family Comparison. Helps diagnose whether your model class is adequate and whether there09s remaining learnable signal",
        "Nonlinear >> linear b There is real nonlinear structure not captured by simple models.",
        "Linear >> nonlinear b Tree model overfitting.",
        "All models perform similarly well b The problem is stable and well-posed.",
        "All models perform similarly poorly b The target has low intrinsic predictability.",
        "Ensembles significantly better b High model uncertainty; more data helps",
    ])

    combined_payload = {
        "cv_splits_learn": int(cv_splits_learn),
        "cv_splits_robust": int(cv_splits_robust),
        "label_quality": lq_payload,
        "label_learnability": ll_payload,
        "model_robustness": mr_payload,
    }

    json_path, md_path = _export_report(
        prefix="snr_full_diagnostics",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=combined_payload,
        markdown_lines=md_lines,
    )

    print(f"\nFull diagnostics report saved to: {json_path} and {md_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--symbol", type=str, default="ETHUSDT")
    sub.add_argument("--exchange", type=str, default="binance")
    sub.add_argument("--timeframe", type=str, default="15m")
    sub.add_argument("--direction", type=str, default="long", choices=["long", "short", "both"])
    sub.add_argument("--model", type=str, default="analyst")


def main() -> None:
    parser = argparse.ArgumentParser(description="SNR and label diagnostics for meta-labeling outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # label-quality
    p_quality = subparsers.add_parser("label-quality", help="Label distribution and economic SNR diagnostics")
    _add_common_args(p_quality)

    # label-learnability
    p_learn = subparsers.add_parser("label-learnability", help="Learnability and entropy-based label quality")
    _add_common_args(p_learn)
    p_learn.add_argument("--cv-splits", type=int, default=3)

    # model-robustness
    p_robust = subparsers.add_parser("model-robustness", help="Probe model CV robustness diagnostics")
    _add_common_args(p_robust)
    p_robust.add_argument("--cv-splits", type=int, default=5)

    # full
    p_full = subparsers.add_parser("full", help="Run all diagnostics and aggregate results")
    _add_common_args(p_full)
    p_full.add_argument("--cv-splits-learn", type=int, default=3)
    p_full.add_argument("--cv-splits-robust", type=int, default=5)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.command == "label-quality":
        run_label_quality(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
        )

    elif args.command == "label-learnability":
        run_label_learnability(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            cv_splits=args.cv_splits,
        )

    elif args.command == "model-robustness":
        run_model_robustness(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            cv_splits=args.cv_splits,
        )

    elif args.command == "full":
        run_full(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            model=args.model,
            cv_splits_learn=args.cv_splits_learn,
            cv_splits_robust=args.cv_splits_robust,
        )


if __name__ == "__main__":
    main()
