# Retention-feature substrate contract

This contract defines the minimum new data required before reopening Stage B
for entry-time `retain | clear` modelling.  It does not authorise blending
these labels into entry EV yet.

## Required row identity

Every feature snapshot must join one-to-one to the candidate contract by:

`candidate_id, symbol, side, decision_ts, feature_cutoff_ts`.

`feature_cutoff_ts <= decision_ts` is a hard assertion.  A row is rejected,
not imputed from a later snapshot, when a required source is stale or absent.

## Required factual features

At the decision timestamp, materialise both level and causal changes over
predeclared 1m/5m/15m windows:

- top-of-book imbalance and bid/ask depth at 10/25/50 bps;
- depth change, replenishment/resilience after a sweep, and spread change;
- signed/aggressor trade flow, volume imbalance, and flow acceleration;
- liquidation volume/impulse and open-interest change;
- distance to contemporaneous high-liquidity clusters or liquidation bands;
- continuation/exhaustion composites of trend, volatility expansion, OI,
  breadth, flow, and liquidity resilience.

No value may use any book, trade, funding, OI, liquidation, or price update
after `decision_ts`.

## Separate provenance fields

Each feature must carry source timestamp range, exchange/product mapping,
source completeness, and a feature-calculation version.  Historical L2/trade
sources cannot be replaced by current/live snapshots or a cross-sectional
proxy without changing the feature ID and marking it as a distinct arm.

## Required validation before entry modelling

1. Coverage: report full candidate denominator, missing/stale fraction by
   month, symbol and side; require a predeclared common cohort.
2. Point in time: assert max source timestamp is no later than cutoff.
3. Strict OOF `retain | clear` diagnostic: report AUC, Brier, log loss,
   calibration and month/side transport before any EV combination.
4. Incremental test: compare the frozen Stage-B hierarchy with and without
   the new OOF prediction/features using identical candidates and policy.
5. Admission: retain only if global top-tail, causal-threshold and latest
   month economics improve without a side failure.  Otherwise keep it
   diagnostic/action-layer only.

## Explicit exclusions

Do not use realised post-entry MFE/MAE, future order book, eventual exit,
future funding/OI, future liquidation path, or retrospectively selected
liquidity clusters.  Do not promote results from the current candidate-
conditioned/current-spread-counterfactual panel.
