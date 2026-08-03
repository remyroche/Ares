# Stage-D compact action model v9 deterministic audit

## Canonical and verification artifacts

- Canonical: `data_perp/artifacts/stage_d_compact_action_model_20260731_v9`
- Same-code verification rerun: `data_perp/artifacts/stage_d_compact_action_model_20260731_v10`
- Result: all 11 files are byte-identical, including `run_manifest.json` and `manifest.sha256`.
- v7/v8 are superseded because their multi-thread LightGBM runs failed byte identity.
- Canonical runner SHA-256: `2d0e4005f7bbbb5fc7316ec6dd6a08b7c24956a2087208838bf078483e511ba3`
- Canonical tests SHA-256: `41100ccddc51995ee23fd7e5e320c9a42e8ed749e8b971f0ca33fd47ffec2638`
- Focused tests: 10 passed.

## Corrected feature dependency

- Bound only to canonical feature pack `stage_d_action_features_20260731_v5`.
- Feature v5/v6 same-code verification is 11/11 byte-identical.
- Feature run-manifest SHA-256: `e9c32a5f2da845b9b635701ec60360e8d08371de3f1eea2856c962f5f244dd46`.
- Feature parquet SHA-256: `a953a255f233ae210039fc8fb56a5c2f78a713f3b3b466cca81a4746c20cacd5`.
- Feature lineage SHA-256: `178e211acce4fe0620732e3797a3ed350e6544a6720950474345f4c01f97c151`.

## Compact development-only re-admission

D2 approved A1, but A1 did not survive the compact re-admission rule. On identical development OOF rows, A0-only improved all required metrics while preserving calibration:

- policy net: 94.310198 vs 94.167751 bps;
- MAE: 134.090580 vs 135.073639 bps;
- Spearman IC: 0.802503 vs 0.795197;
- calibration error: 8.57e-13 vs 9.55e-12.

The frozen compact model therefore uses A0 only. Final OOS was descriptive and could not select the group or margin.

## Final identical-row evidence

- Rows: 31,258; margin selected on development OOF: 0 bps.
- A0-only policy net: 104.849168 bps/trade.
- Increment vs always continue: +80.122798 bps/trade.
- Increment vs always exit: +98.616308 bps/trade.
- Long increment vs continue: +90.448584 bps/trade.
- Short increment vs continue: +68.487407 bps/trade.
- Latest month (2024-11) increment vs continue: +88.971227 bps/trade.
- Day-block bootstrap CI vs continue: [75.845017, 84.670889] bps; probability positive 1.0.
- Symbol support: 126; maximum absolute symbol uplift concentration: 0.015832.
- Full A0+A1 and leave-A1-out final LOO use identical candidate rows; the A0-only final model is also slightly better than full A0+A1 on policy net, MAE, and IC.

All eight research gates pass, including the evidence-derived lineage gate. Terminal status is `CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES`; entry and portfolio policy remain unchanged.
