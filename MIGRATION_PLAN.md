# Migration Plan: Research Framework Reorganization

## 🎯 **Migration Overview**

Consolidating 3 overlapping research directories into 1 coherent framework:

```
BEFORE:
src/research/
├── price_patterns/          (490 pattern matches)
├── mixed_factor_analysis/   (396 pattern matches)  
└── clusters/               (928 pattern matches)

AFTER:
src/research/cluster_analysis/
├── price_patterns/         # Pattern discovery & definition
├── market_factor_analysis/ # Feature → dimensions transformation  
├── clustering/             # Market state discovery
└── economic_relevance/     # Dimension-pattern relationships
```

## 📋 **Detailed Migration Steps**

### **Phase 1: Price Patterns** (`price_patterns/`)

#### **Source Files to Migrate:**
```
FROM: src/research/price_patterns/
├── core_patterns.py                    → mathematical_definitions.py
├── pure_price_action_patterns.py       → pure_price_patterns.py  
├── gradient_targets.py                  → MERGE into mathematical_definitions.py
├── lstm_discovery.py                    → ml_discovery/lstm_discovery.py
├── matrix_profile_discovery.py          → ml_discovery/matrix_profile_discovery.py
├── ml_pure_price_pattern_discovery.py   → ml_discovery/clustering_discovery.py
└── pattern_discovery_framework.py       → CONSOLIDATE with above

FROM: src/research/mixed_factor_analysis/
├── pattern_ml_integration.py            → MERGE mathematical definitions
└── ml_pattern_discovery.py              → ml_discovery/anomaly_discovery.py
```

#### **Key Consolidations:**
1. **Mathematical Definitions**: Merge pure price patterns with ML integration patterns
2. **ML Discovery**: Consolidate 4 different ML discovery approaches
3. **Validation**: Unify pattern validation across approaches

#### **Migration Commands:**
```bash
# Create base files
cp src/research/price_patterns/core_patterns.py \
   src/research/cluster_analysis/price_patterns/mathematical_definitions.py

cp src/research/price_patterns/pure_price_action_patterns.py \
   src/research/cluster_analysis/price_patterns/pure_price_patterns.py

# Create ML discovery modules
cp src/research/price_patterns/lstm_discovery.py \
   src/research/cluster_analysis/price_patterns/ml_discovery/lstm_discovery.py

cp src/research/price_patterns/matrix_profile_discovery.py \
   src/research/cluster_analysis/price_patterns/ml_discovery/matrix_profile_discovery.py

# Create validation module
touch src/research/cluster_analysis/price_patterns/pattern_validation.py
```

### **Phase 2: Market Factor Analysis** (`market_factor_analysis/`)

#### **Source Files to Migrate:**
```
FROM: src/research/clusters/
├── dimension_analyzer.py                → dimension_discovery.py
├── advanced_feature_engineering.py      → factor_extraction.py
├── statistical_dimension_analysis.py    → statistical_analysis.py

FROM: src/research/mixed_factor_analysis/
├── (statistical components)             → MERGE into statistical_analysis.py
```

#### **Key Consolidations:**
1. **Dimension Discovery**: Unify implicit dimension discovery methods
2. **Statistical Analysis**: Consolidate PCA, FA, ICA approaches
3. **Feature Clustering**: Group features into coherent dimensions

#### **Migration Commands:**
```bash
# Create base files
cp src/research/clusters/dimension_analyzer.py \
   src/research/cluster_analysis/market_factor_analysis/dimension_discovery.py

cp src/research/clusters/advanced_feature_engineering.py \
   src/research/cluster_analysis/market_factor_analysis/factor_extraction.py

cp src/research/clusters/statistical_dimension_analysis.py \
   src/research/cluster_analysis/market_factor_analysis/statistical_analysis.py

# Create feature clustering module
touch src/research/cluster_analysis/market_factor_analysis/feature_clustering.py
```

### **Phase 3: Clustering** (`clustering/`)

#### **Source Files to Migrate:**
```
FROM: src/research/clusters/
├── regime_clusterer.py                  → regime_discovery.py
├── similarity_matrix_clustering.py      → similarity_clustering.py
├── validation_metrics.py                → validation_metrics.py
├── data_driven_clustering_framework.py  → optimal_cluster_selection.py
├── adaptive_clustering.py               → MERGE into regime_discovery.py
```

#### **Key Consolidations:**
1. **Regime Discovery**: Unify clustering approaches
2. **Validation**: Comprehensive cluster validation metrics
3. **Optimal Selection**: Data-driven cluster number selection

#### **Migration Commands:**
```bash
# Create base files
cp src/research/clusters/regime_clusterer.py \
   src/research/cluster_analysis/clustering/regime_discovery.py

cp src/research/clusters/similarity_matrix_clustering.py \
   src/research/cluster_analysis/clustering/similarity_clustering.py

cp src/research/clusters/validation_metrics.py \
   src/research/cluster_analysis/clustering/validation_metrics.py

cp src/research/clusters/data_driven_clustering_framework.py \
   src/research/cluster_analysis/clustering/optimal_cluster_selection.py
```

### **Phase 4: Economic Relevance** (`economic_relevance/`)

#### **Source Files to Migrate:**
```
FROM: src/research/mixed_factor_analysis/
├── economic_relevance_research_framework.py → causal_analysis.py
├── pattern_ml_integration.py                → pattern_dimension_analysis.py

FROM: src/research/clusters/
├── dimension_economic_relevance.py          → MERGE into pattern_dimension_analysis.py
├── economic_metrics.py                      → trading_significance.py
```

#### **Key Consolidations:**
1. **Pattern-Dimension Analysis**: Which dimensions predict which patterns
2. **Causal Analysis**: Granger causality, instrumental variables
3. **Trading Significance**: Economic value measurement

#### **Migration Commands:**
```bash
# Create base files
cp src/research/mixed_factor_analysis/economic_relevance_research_framework.py \
   src/research/cluster_analysis/economic_relevance/causal_analysis.py

cp src/research/mixed_factor_analysis/pattern_ml_integration.py \
   src/research/cluster_analysis/economic_relevance/pattern_dimension_analysis.py

cp src/research/clusters/economic_metrics.py \
   src/research/cluster_analysis/economic_relevance/trading_significance.py

# Create market state relevance module
touch src/research/cluster_analysis/economic_relevance/market_state_relevance.py
```

## 🔧 **Code Consolidation Tasks**

### **High Priority Merges:**

1. **Pattern Definitions** (HIGH redundancy):
```python
# MERGE: Multiple pattern definition approaches
# - core_patterns.py (mathematical approach)
# - pattern_ml_integration.py (ML integration approach)  
# - pure_price_action_patterns.py (pure price approach)
# INTO: mathematical_definitions.py (unified approach)
```

2. **Economic Relevance** (HIGH redundancy):
```python
# MERGE: Multiple economic analysis approaches
# - economic_relevance_research_framework.py (comprehensive)
# - dimension_economic_relevance.py (regime-focused)
# INTO: pattern_dimension_analysis.py + causal_analysis.py
```

3. **ML Discovery** (MEDIUM redundancy):
```python
# CONSOLIDATE: Multiple ML discovery methods
# - lstm_discovery.py
# - matrix_profile_discovery.py  
# - ml_pure_price_pattern_discovery.py
# - ml_pattern_discovery.py (from mixed_factor_analysis)
# INTO: Separate specialized modules in ml_discovery/
```

## 📊 **Expected Outcomes**

### **Before Migration:**
- **3 directories** with overlapping functionality
- **~60% code redundancy** across directories
- **Inconsistent interfaces** between components
- **Unclear research workflow**

### **After Migration:**
- **1 coherent framework** with clear workflow
- **~15% code redundancy** (only necessary duplication)
- **Consistent interfaces** across all components  
- **Clear research progression**: Features → Dimensions → States → Relevance

## ⚠️ **Migration Risks & Mitigation**

### **High Risk:**
1. **Import Dependencies**: Many files import from each other
   - **Mitigation**: Update all import statements systematically
   - **Tool**: Use grep to find all imports, update in phases

2. **Configuration Consistency**: Different config approaches
   - **Mitigation**: Standardize configuration classes
   - **Tool**: Create unified config system first

### **Medium Risk:**
3. **API Compatibility**: Existing code may depend on old structure
   - **Mitigation**: Create compatibility wrappers initially
   - **Tool**: Gradual deprecation of old interfaces

4. **Test Coverage**: Tests may break with restructuring
   - **Mitigation**: Update tests alongside migration
   - **Tool**: Run tests after each migration phase

## 🚀 **Implementation Timeline**

### **Week 1: Setup & Phase 1**
- Create directory structure ✅
- Migrate price patterns
- Create unified pattern definitions
- Update imports and tests

### **Week 2: Phase 2 & 3**  
- Migrate market factor analysis
- Migrate clustering components
- Test integration between phases

### **Week 3: Phase 4 & Integration**
- Migrate economic relevance
- Create unified framework interfaces
- Integration testing

### **Week 4: Cleanup & Documentation**
- Remove old directories
- Update all documentation
- Create migration guides for existing code

## 📋 **Success Criteria**

1. **Single coherent framework** with clear 4-step workflow
2. **Reduced code redundancy** from 60% to <15%
3. **Consistent APIs** across all components
4. **Comprehensive documentation** for each phase
5. **Backward compatibility** for existing integrations
6. **Full test coverage** for migrated components

---

**Migration Lead**: Research Team  
**Timeline**: 4 weeks  
**Priority**: High (reduces technical debt significantly)