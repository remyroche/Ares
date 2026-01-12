class Layer2Error(Exception):
    """Base exception for Layer 2 pipeline."""
    pass

class Layer2RecoverableError(Layer2Error):
    """Error that can be handled gracefully (e.g., skip geometry)."""
    pass

class Layer2FatalError(Layer2Error):
    """Critical error requiring pipeline abort."""
    pass
