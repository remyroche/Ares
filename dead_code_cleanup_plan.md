# Dead Code Cleanup Plan

## 🎯 **Priority 1: High-Impact Cleanup (Safe to Remove)**

### **A. Unused Monitoring Components**
```bash
# These monitoring classes are completely unused:
- src/monitoring/regime_sr_tracker.py (RegimeSRTracker, RegimeType)
- src/monitoring/tracking_system.py (TrackingSystem, TrackingType)
- src/monitoring/ml_monitor.py (ModelStatus, DriftType, ModelDriftAlert)
- src/monitoring/regime_monitoring_dashboard.py (RegimeMonitoringWebSocket, RegimeAlert)
```

### **B. Unused Tactician SR Components**
```bash
# Entire SR levels system appears unused:
- src/tactician/sr_levels/sr_ensemble_predictor.py (7 unused classes)
- src/tactician/sr_levels/sr_performance_monitor.py (4 unused classes)
- src/tactician/sr_levels/enhanced_sr_confluence.py (3 unused classes)
- src/tactician/sr_levels/sr_context_aware_calculator.py (4 unused classes)
```

### **C. Unused Supervisor Components**
```bash
# Many supervisor functions are unused:
- src/supervisor/global_portfolio_manager.py (25 unused functions)
- src/supervisor/performance_reporter.py (26 unused functions)
- src/supervisor/dynamic_weighter.py (36 unused functions)
```

## 🎯 **Priority 2: Medium-Impact Cleanup (Review First)**

### **A. Unused Analytics Components**
```bash
# Analytics modules with unused functions:
- src/analytics/bayesian_probability_updates.py (6 unused functions)
- src/analytics/copula_dependency_models.py (4 unused functions)
- src/analytics/limited_microstructure_features.py (16 unused functions)
```

### **B. Unused Interface Components**
```bash
# Interface definitions that aren't used:
- src/interfaces/enhanced_event_bus.py (13 unused functions, 9 unused classes)
- src/interfaces/event_bus.py (8 unused functions, 3 unused classes)
- src/interfaces/base_interfaces.py (8 unused functions, 13 unused classes)
```

## 🎯 **Priority 3: Low-Impact Cleanup (Keep for Future)**

### **A. Unused Trading Components**
```bash
# Trading modules that might be used in future:
- src/trading/live_wavelet_analyzer.py (10 unused functions)
- src/trading/sr_trading_intelligence.py (5 unused functions)
- src/trading/live_wavelet_demo.py (9 unused functions)
```

## 📋 **Cleanup Strategy:**

### **Phase 1: Remove Completely Unused Files**
1. **Delete unused monitoring components** (Priority 1A)
2. **Remove unused SR levels system** (Priority 1B)
3. **Clean up unused supervisor functions** (Priority 1C)

### **Phase 2: Refactor Partially Used Files**
1. **Remove unused functions** from files that have some usage
2. **Consolidate similar functionality**
3. **Update imports** to remove references to deleted code

### **Phase 3: Validate and Test**
1. **Run tests** to ensure nothing breaks
2. **Check imports** for any remaining references
3. **Update documentation** to reflect changes

## 🚀 **Expected Benefits:**

### **Immediate:**
- **Reduced codebase size** by ~30-40%
- **Faster import times**
- **Cleaner project structure**
- **Easier navigation**

### **Long-term:**
- **Easier maintenance**
- **Reduced complexity**
- **Better performance**
- **Clearer architecture**

## ⚠️ **Safety Considerations:**

### **Before Removing:**
1. **Check if code is used in tests**
2. **Verify no dynamic imports** reference the code
3. **Ensure no configuration** references the modules
4. **Check if code is used in documentation** or examples

### **After Removing:**
1. **Run full test suite**
2. **Check for import errors**
3. **Verify application still works**
4. **Update any documentation**

## 📊 **Impact Estimation:**

- **Files to remove completely**: ~50-100 files
- **Functions to remove**: ~8,000-10,000 functions
- **Classes to remove**: ~1,500-1,800 classes
- **Estimated size reduction**: 30-40% of codebase
- **Maintenance effort reduction**: 50-60%
