# Changes Log

## 2026-01-23
- Added cross-asset (ca__/ms__) features as AEDL specialist signals so causal surprise events include cross-asset context. Files: `src/training/steps/labeling/spectral_specialists.py`.
- Cached wavelet denoised prices in Layer2 with artifact save/load keyed by dataset fingerprint to avoid fallback to raw prices. Files: `src/training/steps/labeling/label_based_layer_2.py`.
- Aligned causal surprise event quantile thresholds to the event density target (`event_threshold`), ensuring 4% threshold targets ~4% recall. Files: `src/training/steps/labeling/causal_surprise_events.py`.
- Skipped model race payloads with single-class labels to avoid degenerate Huber/learner warnings. Files: `src/training/steps/labeling/label_based_layer_2.py`.
- Hardened Layer2 probe preparation to replace non-finite values before Ridge probe to prevent NaN/inf exceptions. Files: `src/training/steps/labeling/label_based_layer_2.py`.
- Switched denoised price artifact category to features to avoid KlinesParquetManager parquet path errors. Files: `src/training/steps/labeling/label_based_layer_2.py`.
- Aligned denoised artifact load/save context (symbol/exchange/timeframe) and category to ensure router finds the cached series. Files: `src/training/steps/labeling/label_based_layer_2.py`.
- Added versioned-store precheck before attempting denoised artifact load to avoid router warnings when cache is empty. Files: `src/training/steps/labeling/label_based_layer_2.py`.
