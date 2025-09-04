"""Configuration for enhanced prediction service."""

def get_enhanced_prediction_service_config():
    """Get configuration for enhanced prediction service."""
    return {
        'model_path': 'models/',
        'confidence_threshold': 0.7,
        'fallback_enabled': True
    }