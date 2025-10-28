# Phase 3 & Feature Classification - Documentation Index

## 📚 Complete Documentation Suite

This index helps you navigate all the documentation created for understanding the feature generation pipeline, Phase 3, and feature classification.

---

## 🎯 Quick Start

**New to the pipeline?** Read in this order:
1. Start with `COMPLETE_PHASE3_SUMMARY.md` (5 min read)
2. Then read `FEATURE_FLOW_EXPLANATION.md` (10 min read)
3. For deep dive, read `PHASE3_DETAILED_EXPLANATION.md` (20 min read)

**Just need classification fix?** Read:
- `FINAL_CLASSIFICATION_SUMMARY.md` (5 min read)

---

## 📖 Documentation Files

### 1. **COMPLETE_PHASE3_SUMMARY.md** ⭐ START HERE
**Purpose**: Quick reference for Phase 3 concepts

**What you'll learn**:
- What Phase 3 does in 3 steps (3.1, 3.2, 3.3)
- How final_features are selected (80 features)
- How interactions are discovered (20-50 features)
- Real-world example with actual feature names
- Why both final_features AND interactions matter

**Best for**: Understanding Phase 3 at a high level

**Reading time**: ~5 minutes

---

### 2. **FEATURE_FLOW_EXPLANATION.md** ⭐ COMPREHENSIVE GUIDE
**Purpose**: Complete pipeline flow from start to finish

**What you'll learn**:
- All 5 phases (Phase 0 through Phase 4)
- How features transform at each phase
- Input/output at each stage
- Feature expansion: 40 → 800 → 400 → 80 + 50 = 130
- Classification logic (5 categories)
- What gets saved and why

**Best for**: Understanding the entire pipeline end-to-end

**Reading time**: ~10-15 minutes

---

### 3. **PHASE3_DETAILED_EXPLANATION.md** 🔬 TECHNICAL DEEP DIVE
**Purpose**: Complete technical details of Phase 3

**What you'll learn**:
- Exact LGBM parameters for each sub-phase
- How composite scoring works (60/40, 60/30/10 splits)
- How tree analysis extracts feature pairs
- How RFE progressively refines interactions (400→268→180→121→81→54→50)
- Why 5 different scoring methods (MI, Redundancy, LGBM, SHAP, Stability)
- How each operation is computed (_x_, _div_, _minus_, _log_, _log_ratio_)
- Performance timing breakdown

**Best for**: Deep technical understanding, debugging, optimization

**Reading time**: ~20-30 minutes

---

### 4. **FINAL_CLASSIFICATION_SUMMARY.md** ✅ CLASSIFICATION FIX
**Purpose**: Summary of the feature classification fix

**What you'll learn**:
- The 5 feature categories (with visual flowchart!)
- Why `_base` is a BASE feature, not a variant
- Priority order of classification checks
- Complete examples for each category
- Code changes made
- Validation test results

**Best for**: Understanding feature categorization logic

**Reading time**: ~5 minutes

---

### 5. **FEATURE_CLASSIFICATION_FIX_SUMMARY.md** 🔧 TECHNICAL FIX DETAILS
**Purpose**: Technical details of the classification fix

**What you'll learn**:
- Original issue description
- Code location and changes
- Before/after comparison
- Testing methodology

**Best for**: Code review, understanding what changed

**Reading time**: ~3 minutes

---

## 🎓 Learning Paths

### Path 1: "I just want to understand Phase 3"
```
1. COMPLETE_PHASE3_SUMMARY.md         (5 min)
2. PHASE3_DETAILED_EXPLANATION.md     (20 min) [optional, if you want details]
```
**Total**: 5-25 minutes

---

### Path 2: "I need to understand the whole pipeline"
```
1. COMPLETE_PHASE3_SUMMARY.md         (5 min)  [Phase 3 overview]
2. FEATURE_FLOW_EXPLANATION.md        (15 min) [Complete flow]
3. FINAL_CLASSIFICATION_SUMMARY.md    (5 min)  [Classification]
```
**Total**: 25 minutes

---

### Path 3: "I'm debugging or optimizing"
```
1. FEATURE_FLOW_EXPLANATION.md        (15 min) [Complete flow]
2. PHASE3_DETAILED_EXPLANATION.md     (30 min) [Deep technical dive]
3. FINAL_CLASSIFICATION_SUMMARY.md    (5 min)  [Classification rules]
```
**Total**: 50 minutes

---

### Path 4: "I just need to know about the classification fix"
```
1. FINAL_CLASSIFICATION_SUMMARY.md           (5 min)
2. FEATURE_CLASSIFICATION_FIX_SUMMARY.md     (3 min) [optional details]
```
**Total**: 5-8 minutes

---

## 🔑 Key Concepts Quick Reference

### final_features
- **What**: 80 best individual features from Phase 3.2
- **Examples**: `rsi_base`, `macd_volnorm`, `volume_vwap`, `rsi_base_3x_ratio`
- **Think**: "Solo performers"

### interactions  
- **What**: 20-50 synergistic feature combinations from Phase 3.3
- **Examples**: `rsi_base_x_macd_volnorm`, `volume_div_atr`, `rsi_3x_ratio_x_macd_6x_ratio`
- **Think**: "Duets"

### Hybrid CT Interactions
- **What**: Interactions between cross-timeframe features
- **Example**: `rsi_base_3x_ratio_x_macd_volnorm_6x_ratio`
- **Think**: "Most sophisticated features - combining timeframes + variants + interactions"

### The 5 Feature Categories
1. **Hybrid CT interactions**: Has interaction operator + CT marker
2. **Traditional interactions**: Has interaction operator only
3. **Cross-timeframe ratios**: Has CT marker only
4. **Variant features**: Ends with `_volnorm`, `_vwap`, or `_trend_adj`
5. **Base features**: Everything else (including `_base` suffix!)

---

## 📊 Pipeline at a Glance

```
Phase 0: Load & Select
  Input: Feature bank
  Output: Top N per category (e.g., 40 features)
  
Phase 1: Variants + Cross-Timeframe
  Input: 40 features
  Output: 800 features (40 × 4 variants × 5 timeframes)
  
Phase 2: Cheap Pruning
  Input: 800 features
  Output: 400-480 features (40-50% pruning)
  
Phase 3: LGBM+SHAP Pipeline ⭐ MOST IMPORTANT
  Input: 400-480 features
  Phase 3.1: 400 → 120 (fast filtering)
  Phase 3.2: 120 → 80 (accurate selection) → final_features
  Phase 3.3: 80 → 80 interactions → interactions
  Output: 80 final_features + 80 interactions
  
Phase 4: Combine & Save
  Input: final_features + interactions
  Output: combined_features (160 total)
  Action: Classify into 5 categories and save
```

---

## 🎯 Common Questions

### Q: What's the difference between final_features and interactions?
**A**: 
- `final_features` = 80 best individual features
- `interactions` = 20-50 combinations of those features
- See `COMPLETE_PHASE3_SUMMARY.md` section "Why Both?"

### Q: How are interactions discovered?
**A**: 
- Train LGBM on final_features
- Extract feature pairs from tree co-occurrence
- Generate 5 operations per pair (×, ÷, -, log, log_ratio)
- Score with 5 methods using RFE
- See `PHASE3_DETAILED_EXPLANATION.md` for full details

### Q: What are hybrid CT interactions?
**A**:
- Interactions between cross-timeframe features
- Example: `rsi_3x_ratio_x_macd_6x_ratio`
- Most sophisticated feature type
- See `FINAL_CLASSIFICATION_SUMMARY.md` for examples

### Q: Why is `_base` a base feature and not a variant?
**A**:
- `_base` is the base/original version after robust scaling
- Variants are transformations: `_volnorm`, `_vwap`, `_trend_adj`
- See `FINAL_CLASSIFICATION_SUMMARY.md` Rule 1

### Q: How long does Phase 3 take?
**A**:
- Phase 3.1: 2-3 minutes
- Phase 3.2: 3-5 minutes
- Phase 3.3: 10-15 minutes
- Total: 15-23 minutes (~70% of pipeline time)
- See `PHASE3_DETAILED_EXPLANATION.md` Performance Stats

---

## 🔧 Code Reference

### Main File
`src/training/steps/pre_training/feature_generation_interaction_generation_step.py`

### Key Functions
- `_phase3_1_shallow_sweep()`: Lines 2315-2614
- `_phase3_2_deeper_refinement()`: Lines 2795-3052
- `_phase3_3_interaction_discovery()`: Lines 3053-3460
- Feature classification: Lines 3611-3650

### Classification Logic
```python
# Lines 3618-3619
variant_suffixes = ['_volnorm', '_vwap', '_trend_adj']  # NOT _base!

# Lines 3621-3636
Priority:
1. Check interaction operators → Hybrid or Traditional
2. Check CT markers → CT ratio
3. Check variant suffixes → Variant or Base
```

---

## ✅ Quick Validation

To verify you understand the concepts, can you answer these?

1. How many features does Phase 3.1 output? **Answer**: 120 features
2. How many features does Phase 3.2 output? **Answer**: 80 (final_features)
3. How many operations are applied to each feature pair? **Answer**: 5 (_x_, _div_, _minus_, _log_, _log_ratio_)
3. Is `rsi_base` a base or variant feature? **Answer**: Base
4. What's a hybrid CT interaction? **Answer**: Interaction with CT markers (e.g., `rsi_3x_ratio_x_macd_6x_ratio`)
5. How many scoring methods in Phase 3.3 RFE? **Answer**: 5 (MI, Redundancy, LGBM, SHAP, Stability)

---

## 📞 Support

If something is unclear:
1. Check the relevant documentation file from the list above
2. Use the learning paths to build understanding progressively
3. Start with summaries, dive into details only when needed

---

## 🎉 Summary

This documentation suite explains:
- ✅ The complete feature generation pipeline (5 phases)
- ✅ Phase 3 in detail (how features are selected and interactions discovered)
- ✅ Feature classification (5 categories with proper `_base` handling)
- ✅ Why the pipeline works this way (design decisions)

**Most important takeaway**: Phase 3 produces 80 "solo performers" (final_features) and 20-50 "duets" (interactions) using sophisticated multi-criteria selection and RFE-based interaction discovery. Together with proper 5-way classification, you get a comprehensive, high-quality feature set for model training.
