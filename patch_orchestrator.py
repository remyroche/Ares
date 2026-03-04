import re

with open('extreme_price_movements/position_sizer/training_orchestrator.py', 'r') as f:
    content = f.read()

search = """def _quantile_metrics(y_true, q50, qh, qh_level: float) -> dict:
    y = np.asarray(y_true, dtype=float)
    p50 = np.asarray(q50, dtype=float)
    ph = np.asarray(qh, dtype=float)
    m = np.isfinite(y) & np.isfinite(p50) & np.isfinite(ph)
    if not np.any(m):
        return {
            "n": 0,
            "pinball_q50": float("nan"),
            "pinball_qh": float("nan"),
            "coverage_q50": float("nan"),
            "coverage_qh": float("nan"),
            "mean_y": float("nan"),
            "mean_q50": float("nan"),
            "mean_qh": float("nan"),
        }
    yy = y[m]
    p50m = p50[m]
    phm = ph[m]
    return {
        "n": int(len(yy)),
        "pinball_q50": _pinball_loss(yy, p50m, 0.50),
        "pinball_qh": _pinball_loss(yy, phm, float(qh_level)),
        "coverage_q50": float(np.mean(yy <= p50m)),
        "coverage_qh": float(np.mean(yy <= phm)),
        "mean_y": float(np.mean(yy)),
        "mean_q50": float(np.mean(p50m)),
        "mean_qh": float(np.mean(phm)),
    }"""

replace = """def _quantile_metrics(y_true, q50, qh, qh_level: float) -> dict:
    y = np.asarray(y_true, dtype=float)
    p50 = np.asarray(q50, dtype=float)
    ph = np.asarray(qh, dtype=float)
    m = np.isfinite(y) & np.isfinite(p50) & np.isfinite(ph)
    if not np.any(m):
        return {
            "n": 0,
            "pinball_q50": float("nan"),
            "pinball_qh": float("nan"),
            "coverage_q50": float("nan"),
            "coverage_qh": float("nan"),
            "interval_evaluation": float("nan"),
            "mean_y": float("nan"),
            "mean_q50": float("nan"),
            "mean_qh": float("nan"),
        }
    yy = y[m]
    p50m = p50[m]
    phm = ph[m]

    # Interval Evaluation (Winkler Score or similar, here we use mean risk band width and coverage)
    # The risk band is defined by Q50 to Qh.
    # An alternative is standard mean interval width:
    # Here we just output interval_evaluation as mean band size. Can also compute winkler score if desired.
    # A simple Winkler-like interval score for asymmetric band [q50, qh]:
    # It penalizes width + (2/alpha)*distance if outside.
    # Let's just output mean interval width as requested by the simplest interpretation, or specifically standard Winkler.
    # But wait, Q50 to Qh covers (qh_level - 0.5) probability.
    # Let's output "interval_evaluation_width" and "interval_evaluation_coverage"
    interval_width = float(np.mean(phm - p50m))

    return {
        "n": int(len(yy)),
        "pinball_q50": _pinball_loss(yy, p50m, 0.50),
        "pinball_qh": _pinball_loss(yy, phm, float(qh_level)),
        "coverage_q50": float(np.mean(yy <= p50m)),
        "coverage_qh": float(np.mean(yy <= phm)),
        "interval_evaluation": interval_width, # mean risk band width
        "mean_y": float(np.mean(yy)),
        "mean_q50": float(np.mean(p50m)),
        "mean_qh": float(np.mean(phm)),
    }"""

content = content.replace(search, replace)
with open('extreme_price_movements/position_sizer/training_orchestrator.py', 'w') as f:
    f.write(content)
