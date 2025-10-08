# Pre-training Data Contracts

The pre-training pipeline validates its step boundaries using lightweight data contracts. The schemas
summarise the expected structure for the artifacts exchanged between components and provide
actionable diagnostics when validation fails.

## `LabeledDataSchema`

Represents the `multi_horizon_labeling_result` artifact produced by the multi-horizon profit labeler.

| Field | Type | Notes |
| --- | --- | --- |
| `labeled_data`, `labels` | `pd.DataFrame` | Must contain the standardized target columns with matching shapes. |
| `confidence_scores`, `eligibility_masks`, `sigma_payoffs` | optional `pd.DataFrame` | Numeric feature-style matrices validated with the engineered-feature schema. |
| `horizon_weights` | `Dict[str, float]` | Horizon weights used by downstream steps. |
| `target_columns` | `List[str]` | Namespaced target identifiers for downstream use. |
| `validation_results` | `{is_valid: bool, issues: List[str]}` | Downstream compatibility assessment. |
| `metadata` | `Dict[str, Any]` | Enriched labeling metadata persisted with the artifact. |
| `market_data`, `market_data_batches` | optional `pd.DataFrame` or list of frames | Normalized OHLCV inputs backing the labeling run. |

**Example**

```python
validated = validate_multi_horizon_labeling_result(
    artifacts['multi_horizon_labeling_result'],
    context='pre_training.multi_horizon',
)
```

## `FeaturesSchema`

Encapsulates the feature matrices produced by interactive feature generation.

| Field | Type | Notes |
| --- | --- | --- |
| `features` | `pd.DataFrame` | Primary engineered feature matrix with datetime index. |
| `feature_names`, `selected_features` | `List[str]` | Namespaced feature identifiers. |
| `interaction_features`, `cross_timeframe_features` | optional `pd.DataFrame` | Secondary feature blocks validated like the primary matrix. |
| `execution_time`, `memory_usage_mb` | optional numeric | Diagnostic metadata preserved with the artifact. |

**Example**

```python
features_payload = artifacts['interactive_feature_generation_result']
validated_features = validate_feature_artifact(
    features_payload,
    context='pre_training.interactive_features',
)
```

## `SelectionResultSchema`

Describes the final feature selection result that flows into downstream modelling.

| Field | Type | Notes |
| --- | --- | --- |
| `final_features` | `List[str]` | Mandatory list of selected features. |
| `stage_1_features` … `stage_3_features` | optional `List[str]` | Intermediate stage selections. |
| `feature_counts` | `Dict[str, int]` | Counts for each reduction stage. |
| `stage_scores` | `Dict[str, Dict[str, float]]` | Scoring metadata for every stage. |
| `selection_time` | optional float | Runtime tracking for auditability. |
| `is_unsupervised` | optional bool | Indicates unsupervised runs. |

**Example**

```python
selection_payload = {
    'final_features': selection_result.final_features,
    'feature_counts': selection_result.feature_counts,
    'stage_scores': {
        'stage_1': selection_result.stage_1_scores,
        'stage_2': selection_result.stage_2_scores,
        'stage_3': selection_result.stage_3_scores,
        'final': selection_result.final_scores,
    },
}
validate_selection_artifact(selection_payload, context='pre_training.final_selection')
```

---

These validators surface contract violations with structured error messages, allowing the
sub-pipeline to fail fast with actionable feedback when upstream data drifts or becomes malformed.
