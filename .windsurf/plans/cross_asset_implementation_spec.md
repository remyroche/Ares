# Cross-Asset Implementation Specification

This specification provides the crisp implementation layer with contracts, interfaces, gating logic, invariance checks, and acceptance tests for the cross-asset trading system transformation.

## 1. Non-Negotiable Engineering Conventions

### 1.1 Fit/Transform Boundary (Prevents Leakage)

Every component that learns parameters must implement:
```python
fit(train_df, ...) -> self
transform(df, ...) -> df_or_series  
fit_transform(train_df, ...) -> df_or_series
```

**Rules**:
- fit() may only see training windows
- transform() must not alter learned state
- Rolling statistics computed with explicit min_periods and closed='left' semantics

### 1.2 Time Semantics and Alignment

**Definition** (enforced in one place):
- t = feature timestamp
- ret_1 computed as return(t -> t+1)
- Any feature at t must be derivable from data <= t (closed-left rolling windows)

**Hard assertions**:
- No "centered windows"
- No "future shift" features unless explicitly flagged as debugging

### 1.3 Immutable Feature Store Contract

**Structure**: Table with Index (timestamp, ticker), Columns: raw + engineered + labels

**Rule**: No downstream module may overwrite columns in-place without writing to new namespaced columns

## 2. PanelDataProcessor: Concrete Behavior

### 2.1 transform_to_panel()

**Inputs**: Dict[ticker -> df] where each df has timestamp index and at least px

**Outputs**: Panel df with MultiIndex (timestamp, ticker) sorted, deduped

**Key implementation**:
- Enforce monotonic timestamps per ticker
- Harmonize calendars: union calendar then forward-fill only where economically valid (prices), never labels
- Compute log_px, ret_1, vol, dvol with explicit windows

### 2.2 validate_schema()

**Return structured result**:
```python
@dataclass
class SchemaValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
```

**Hard errors**: Missing required columns, Non-unique index, Look-ahead detected
**Warnings**: Sparse VPIN coverage, Short history, Too many NaNs

### 2.3 enforce_prefix_namespacing()

**Enforcement**:
- Engineered features: sa__/ca__/cs__/ms__
- Raw market fields: raw__ prefix
- Labels: y__ prefix (e.g., y__ret_1)

### 2.4 Leakage Sentinels

**Shift Sensitivity Test**: Check correlation with ret_-1 vs ret_+1 for random features
**Timestamp Perturbation Test**: Randomly permute timestamps; predictive power should collapse

## 3. MarketStateVector: Implementation Specification

### 3.1 State Instrument Table

**Input contract**: Wide dataframe indexed by timestamp, columns for state instruments (BTC, ETH, dominance, funding, etc.)

**Rule**: No missing timestamps; if missing, imputed via explicitly documented method

### 3.2 Outputs and Persistence

**Persist (versioned)**: PCA loadings matrix, scaler parameters, state clustering model, rolling similarity history

### 3.3 Stability Guard (Precise)

**Cosine similarity**: Compute component-wise then aggregate:
```python
sim_k(t) = similarity of loading vectors for component k
aggregate = min_k sim_k(t) or weighted min
```

**Acceptance**: min_k sim_k(t) >= 0.9 for at least X% of timestamps in validation window

## 4. CrossAssetSurprises: Required Gating & Outputs

### 4.1 Quantile VPIN Spillover

**Implementation**: statsmodels quantile regression or gradient-boosted quantile model

**Contract**:
- Inputs: panel features slice at time t (no label)
- Outputs: ca__vpin_spill_q50, ca__vpin_spill_q75, etc. (explicit quantiles)
- Separate buy and sell if available

**Acceptance**: For each τ: empirical P(residual <= 0) ≈ τ (within tolerance)

### 4.2 ECT with Tradability Filters

**Formalize**: rolling window length, minimum sample length, half-life estimator method

**Outputs**: ca__ect value, ca__ect_active boolean, ca__ect_half_life, ca__ect_rank_stability

**Critical**: Central gating engine consumes ca__ect_active, not recompute

## 5. MetaModelInvariance: Make the Check Actionable

### 5.1 Environments Definition

**Explicit**: env = ticker for LOAO, env = sector for LOSO, optionally env = ms__state_id

### 5.2 Gradient Alignment Metric (Operational)

**Store**: g_e vectors per env, pairwise cosine distances, summary stats

**Return structure**:
```python
@dataclass
class InvarianceReport:
    dispersion: float
    worst_env_pair: Tuple[str, str]
    worst_distance: float
    per_feature_grad_var: pd.Series
```

### 5.3 Iterative Pruning Protocol (Deterministic)

**Define**: k_drop (e.g., 5 features per iteration), max iterations, stop conditions

**Acceptance**: Dispersion reduced relative to baseline, LOAO/LOSO delta not negative beyond tolerance

## 6. CrossAssetPositionSizer: Percentiles, Entropy, and Top-K

### 6.1 Rolling Calibration (Percentiles)

**Process**:
- Step A (per asset): raw score → calibrated probability via isotonic/platt on rolling window
- Step B (cross-asset): calibrated prob → percentile rank across assets at time t

**Store**: calibration_window_start/end, calibration model per asset

### 6.2 Entropy Filter: Threshold Definition

**Options**: fixed H_thresh or dynamic (median + MAD band over trailing window)

**Output**: gate__entropy_pass boolean

### 6.3 Deterministic Selection Output

**Return table**: timestamp, ticker, score, calibrated_p, percentile, size, gates_passed

## 7. PortfolioConstraints: Tail Correlation Done Correctly

### 7.1 Tail Correlation Definition

**Tail condition**: Timestamps where rm is in bottom 5% of trailing distribution (rolling quantile)

**Acceptance**: Tail constraint binds during stress windows, fallback rule for small tail samples

### 7.2 Beta Exposure

**Measurement**: Versus same market proxy used in MSV/Rm construction

**Outputs**: port__beta_total, port__beta_by_sector, gating flags

## 8. CrossAssetChaser: Residual Learning Protocol

### 8.1 Input/Target Definition

**Target explicit**: err_t = y_true - y_pred_base, chaser predicts err_t

**Outputs**: pred_final = pred_base + chaser_corr

**Acceptance**: Improvement in calibration (Brier) and residual IC, PnL uplift net of turnover

## 9. ValidationBattery: Deterministic, Comparable, Logged

### 9.1 Output Structure

```python
@dataclass
class ValidationResult:
    split_name: str
    metrics: Dict[str, float]     # sharpe, IC, AUC, brier, turnover
    by_asset: Dict[str, Dict[str, float]]
    by_sector: Dict[str, Dict[str, float]]
    artifacts: Dict[str, str]     # paths to plots, serialized reports
```

### 9.2 Baselines

**Required**: single-asset legacy pipeline, pooled non-invariant model (no IRM constraints)

## 10. Central Gating Engine: Missing Class

```python
class GatingEngine:
    def evaluate(self, panel_slice_t: pd.DataFrame, portfolio_state: dict) -> pd.DataFrame:
        """
        Returns per-ticker gate flags and reasons.
        Columns: gate__ect_active, gate__entropy_pass, gate__tail_corr_pass,
                 gate__beta_cap_pass, gate__max_corr_pass, gate__confidence_replace, ...
        """
```

**Requirements**: Human-auditable reason codes, pure function (no hidden state except read-only config)

## 11. Acceptance Test Suite: What to Implement First

### 11.1 Unit Tests (fast)
- Schema enforcement + namespacing
- Rolling window closed-left
- Percentile calibration monotonicity
- Entropy filter deterministic behavior
- Cointegration gating outputs stable

### 11.2 Property Tests (mid)
- Timestamp perturbation collapses predictability
- Synthetic assets degrade gracefully
- LOAO/LOSO split correctness (no overlap)

### 11.3 Integration Tests (slow)
- Full pipeline run produces consistent artifacts
- Reproducibility with fixed seed
- "No trades" behavior when gates fail

## 12. Key Missing Configuration Fields

**Explicitly parameterize in YAML**:
- Rolling windows: returns/vol/beta/cointegration/calibration
- Entropy threshold rule (static vs dynamic)
- Tail quantile (5%) and minimum tail sample size
- MSV update frequency + PCA component count + clustering hyperparams
- IRM invariance target dispersion + pruning protocol params
- Top-K K, sector caps, max simultaneous trades
- Confidence replacement exit rule thresholds

## 13. Recommended Final Adjustment to Class Interfaces

**Changes**:
- Every "compute_*" should accept (df, config) and return column(s) with stable names
- Add GatingEngine class
- Replace bool returns with structured reports for auditability

## 14. Module Specifications

### 14.1 Core Classes Needed

```python
class PanelDataProcessor:
    def transform_to_panel(self, single_asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame
    def validate_schema(self, df: pd.DataFrame) -> SchemaValidationResult
    def enforce_prefix_namespacing(self, df: pd.DataFrame) -> pd.DataFrame
    def detect_leakage(self, df: pd.DataFrame) -> List[str]

class MarketStateVector:
    def compute_state(self, state_instruments: pd.DataFrame) -> pd.DataFrame
    def check_stability(self, loadings_history: List[np.ndarray]) -> bool
    def update_if_needed(self) -> None
    def persist_state(self, version: str) -> None

class CrossAssetSurprises:
    def compute_vpin_spillover(self, panel_df: pd.DataFrame, config: dict) -> pd.DataFrame
    def compute_ect(self, panel_df: pd.DataFrame, config: dict) -> pd.DataFrame
    def validate_activation_conditions(self, panel_df: pd.DataFrame) -> pd.DataFrame

class MetaModelInvariance:
    def compute_gradient_alignment(self, model, environments: List[str]) -> InvarianceReport
    def enforce_no_ticker_id(self, features: pd.DataFrame) -> pd.DataFrame
    def iterative_pruning(self, model, features: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, List[str]]

class CrossAssetPositionSizer:
    def compute_cross_asset_percentiles(self, scores: pd.DataFrame, config: dict) -> pd.DataFrame
    def apply_entropy_filter(self, scores: pd.Series, config: dict) -> pd.Series
    def select_top_k(self, percentiles: pd.DataFrame, k: int) -> pd.DataFrame

class PortfolioConstraints:
    def check_correlation_constraints(self, portfolio: np.ndarray, returns: pd.DataFrame) -> bool
    def check_tail_correlation(self, returns: pd.DataFrame, market_returns: pd.Series, config: dict) -> bool
    def check_beta_exposure(self, betas: np.ndarray, config: dict) -> bool

class CrossAssetChaser:
    def compute_peer_residual_momentum(self, panel_df: pd.DataFrame, config: dict) -> pd.Series
    def compute_relative_volume_clusters(self, panel_df: pd.DataFrame, config: dict) -> pd.Series
    def validate_incremental_value(self, base_predictions: pd.Series, chaser_predictions: pd.Series) -> bool

class ValidationBattery:
    def run_loao_validation(self, model, assets: List[str]) -> ValidationResult
    def run_loso_validation(self, model, sectors: List[str]) -> ValidationResult
    def run_synthetic_asset_test(self, model, synthetic_assets: pd.DataFrame) -> ValidationResult

class GatingEngine:
    def evaluate(self, panel_slice_t: pd.DataFrame, portfolio_state: dict, config: dict) -> pd.DataFrame
    def generate_reason_codes(self, gates: pd.DataFrame) -> pd.DataFrame
```

### 14.2 Data Contracts

```python
@dataclass
class PanelDataSchema:
    timestamp: pd.Timestamp
    ticker: str
    ret_1: float
    px: float
    log_px: float
    vol: float
    dvol: float
    vpin: Optional[float]
    vpin_buy: Optional[float]
    vpin_sell: Optional[float]
    sector: str
    category: str
    # Market state columns (ms__*)
    # Beta columns (beta_*)
    # Features with prefixes (sa__/ca__/cs__/ms__)

@dataclass
class MarketStateConfig:
    state_instruments: List[str]
    n_components: int
    stability_threshold: float
    update_frequency: str
    clustering_method: str

@dataclass
class ValidationConfig:
    loao_assets: List[str]
    loso_sectors: List[str]
    synthetic_asset_configs: List[Dict]
    performance_thresholds: Dict[str, float]
    baseline_comparisons: List[str]

@dataclass
class GatingConfig:
    ect_activation_thresholds: Dict[str, float]
    entropy_threshold_rule: str  # "static" or "dynamic"
    tail_quantile: float
    min_tail_sample: int
    beta_exposure_cap: float
    max_correlation: float
    confidence_replacement_threshold: float
```

This specification provides the complete engineering blueprint with hard contracts, deterministic interfaces, and comprehensive acceptance criteria needed for successful cross-asset implementation.
