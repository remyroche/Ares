# Meta-Labeling HPO Execution and Analysis Plan

Execute the meta_labeling_hpo_sample_weighted pipeline with full monitoring and critical analysis of Layer 2 causal framework results before proceeding to Layer 3.

## Execution Steps

### 1. Run Meta-Labeling HPO Pipeline
Execute the full pipeline with HPO forced:
```bash
python3 src/launcher/ares_launcher.py meta_labeling_hpo_sample_weighted --symbol ETHUSDT --execution-mode full --force-hpo --enable-labeling-hpo
```

### 2. Monitor Execution
- Watch for any bugs or errors during execution
- If issues occur, fix them immediately and restart
- Pay special attention to Layer 2 causal framework integration

### 3. Critical Layer 2 Analysis
Upon completion, thoroughly review Layer 2 results:

**Success Criteria for Layer 2:**
- Multiple geometries selected with Tier 1 + Tier 2 quality
- ORF (Orthogonal Regression Forest) models successfully generated
- ORF models predicting OOF (Out-of-Fold) samples
- Good quality metrics from ML models
- No causal framework violations per de Prado's methodology

**Analysis Sources:**
- Layer 2 diagnostic reports in `outcomes/layer2_gate_diagnostics_*.md`
- Raw metric logs in `outcomes/layer2_raw_metric_log_*.json`
- Geometry gates CSV files
- HPO trial results

**Critical Issues to Check:**
- Causal integrity violations
- Overfitting indicators
- Geometry selection quality
- ORF model performance
- Feature leakage problems

### 4. Layer 3 Conditional Proceed
Only if Layer 2 meets all success criteria, proceed to Layer 3 implementation.

## Key Focus Areas

- **Causal Framework Compliance**: Ensure Layer 2 follows de Prado's causal methodology
- **ORF Model Quality**: Verify orthogonal regression models are properly trained and predicting
- **Geometry Selection**: Check that multiple high-quality geometries are selected
- **Performance Metrics**: Validate ML model metrics are reasonable and not overfit
- **Integration Integrity**: Ensure seamless Layer 2 to Layer 3 transition
