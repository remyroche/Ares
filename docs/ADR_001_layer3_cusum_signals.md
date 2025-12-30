# ADR 001: Injection of CUSUM Signals into Layer 3 Meta-Learner

## Status
Accepted

## Date
2025-12-30

## Context
The Layer 3 (Meta-Labeling) model was observed to be in a "Zombie" state, producing random predictions (AUC ~0.5) and assigning zero importance to ensemble features. This was diagnosed as being caused by a lack of variance in the inputs: the Layer 2 geometries converged to similar parameters, and the Layer 3 model lacked visibility into the underlying market state that triggered the events.

The question arose: **Is it aligned with Lopez de Prado's framework to feed the Primary Model's signal (CUSUM trend/reversal magnitude) directly into the Meta-Model?**

## Decision
We have decided to explicitly compute and inject CUSUM signal values (e.g., `trend_signal`, `reversal_signal`) as features for the Layer 3 Meta-Learner.

## Justification
1.  **Meta-Labeling Alignment:**
    - In *Advances in Financial Machine Learning*, Meta-Labeling (Chapter 3) aims to estimate the probability of success for a given primary signal.
    - The Meta-Model uses features describing the *state of the world* at the time of the trigger to filter false positives.
    - The magnitude/intensity of the primary signal (e.g., "how strong is the trend breakout?") is a critical state variable describing the opportunity. It is not "leakage" because it is known at $t_{entry}$.
    - While the Primary Model uses this signal as a binary trigger (e.g., `signal > threshold`), the Meta-Model can learn non-linear relationships (e.g., "Moderate signals work in low vol, but only Extreme signals work in high vol").

2.  **Bet Sizing Consistency:**
    - Chapter 10 (Bet Sizing) links the probability of success $p$ (output of Meta-Model) to position size.
    - Providing signal magnitude allows the Meta-Model to assign higher $p$ to stronger signals, naturally leading to larger bets for higher-conviction triggers, which aligns with standard discretionary and quantitative trading logic.

3.  **Empirical Verification:**
    - Tests on synthetic data where profitability was a non-linear function of signal strength showed that:
        - **Without CUSUM signals:** Layer 3 AUC ~0.62 (random/weak).
        - **With CUSUM signals:** Layer 3 AUC > 0.99 (learned the pattern).
    - Feature importance analysis confirmed `trend_signal` was a top-ranked feature, proving its utility as a discriminator.

## Consequences
- **Positive:** Layer 3 is no longer a "Zombie" layer; it actively discriminates based on signal quality.
- **Positive:** The system can now learn to avoid "weak breakouts" or "traps" that barely cross the primary threshold.
- **Implementation:** `meta_labeling_hpo_sample_weighted.py` generates these signals if missing, and `label_based_layer_3.py` registers them as candidate features.
