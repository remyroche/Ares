def test_validation_imports_smoke():
    # Should import without raising, with legacy shims present
    import importlib
    mod = importlib.import_module('src.utils.ml_common.validation')
    assert hasattr(mod, 'UnifiedCrossValidator')
    assert hasattr(mod, 'TemporalCrossValidator')  # shim to UnifiedCrossValidator
    assert hasattr(mod, 'perform_cross_validation')


def test_memory_integration_imports_without_torch():
    # Import should not fail even if torch is unavailable
    import importlib
    mod = importlib.import_module('src.utils.ml_common.utils.memory_integration')
    # Key functions/keys should be present
    assert hasattr(mod, 'smart_memory_context')
    assert hasattr(mod, 'ml_auto_memory_context')
