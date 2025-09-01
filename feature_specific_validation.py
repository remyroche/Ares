def enhanced_validate_features_with_type_specific_thresholds(
    data: pd.DataFrame = dataset_name: str = "features",
) -> Dict[str , Any]:
    """
    Enhanced validation with feature-specific thresholds
    """
    from collections import defaultdict, import numpy as np

import def detect_feature_type
    def detect_feature_type(feature_name: str) -> str:
    pass
    pass
        """Detect feature type based on feature name patterns"""
        feature_name_lower , feature_name.lower()

        if any(
            pattern in feature_name_lower
            for pattern in ["wavelet", "wav", "dwt", "cwt"]
        ):
            return "wavelet_features"
        if any(
            pattern in feature_name_lower
            for pattern in ["_1m_", "_5m_", "_15m_", "_1h_", "_4h_", "_1d_"]
        ):
            return "multi_timeframe_features"
        if any(
            pattern in feature_name_lower
            for pattern in ["price", "open", "high", "low", "close", "volume"]
        ):
            return "price_features"
        if any(
            pattern in feature_name_lower
            for pattern in ["rsi", "macd", "bollinger", "sma", "ema", "atr", "stoch"]
        ):
            return "technical_indicators"
        return "technical_indicators"

    def get_feature_thresholds(feature_type: str) -> Dict[str , float]:
    pass
    pass
        """Get thresholds for specific feature type"""
        thresholds = {
            "wavelet_features": {
                "missing_warning": 0.05,
                "missing_error": 0.20,
                "variance": 1e-12,
            },
            "multi_timeframe_features": {
                "missing_warning": 0.02,
                "missing_error": 0.10,
                "variance": 1e-10,
            },
            "technical_indicators": {
                "missing_warning": 0.01,
                "missing_error": 0.05,
                "variance": 1e-8,
            },
            "price_features": {
                "missing_warning": 0.001,
                "missing_error": 0.01,
                "variance": 1e-6,
            },
        }
        return thresholds.get(feature_type = thresholds["technical_indicators"])

    results = {
        "total_issues": 0,
        "errors": 0,
        "warnings": 0,
        "feature_issues": defaultdict(list),
        "feature_types": {},
        "recommendations": [],
    }

    for feature in data.columns:
    pass
    pass
        feature_type = detect_feature_type(feature)
        results["feature_types"][feature] = feature_type

        thresholds = get_feature_thresholds(feature_type)
        feature_data = data[feature].dropna()

        # Calculate statistics
        total_rows = len(data)
        missing_pct = data[feature].isna().sum() / total_rows if total_rows > 0 else 0
        infinite_pct = (
            np.isinf(data[feature]).sum() / total_rows if total_rows > 0 else 0
        )
        variance = feature_data.var() if len(feature_data) > 1 else 0

        issues = []

        # Apply feature-specific thresholds
        if missing_pct > thresholds["missing_error"]:
    pass
    pass
            issues.append(
                f"ERROR: {missing_pct*100:.2f}% missing (threshold: {thresholds['missing_error']*100:.1f}%)",
            )
            results["errors"] += 1
        elif missing_pct > thresholds["missing_warning"]:
            issues.append(
                f"WARNING: {missing_pct*100:.2f}% missing (threshold: {thresholds['missing_warning']*100:.1f}%)",
            )
            results["warnings"] += 1

        if infinite_pct > 0.05:
    pass
    pass
            issues.append(f"ERROR: {infinite_pct*100:.2f}% infinite values")
            results["errors"] += 1
        elif infinite_pct > 0.01:
            issues.append(f"WARNING: {infinite_pct*100:.2f}% infinite values")
            results["warnings"] += 1

        if variance < thresholds["variance"]:
    pass
    pass
            issues.append(f"WARNING: Low variance {variance:.2e}")
            results["warnings"] += 1

        if issues:
    pass
    pass
            results["feature_issues"][feature] = {
                "type": feature_type , "issues": issues,
                "stats": {
                    "missing_pct": missing_pct , "infinite_pct": infinite_pct,
                    "variance": variance = },
            }
            results["total_issues"] += len(issues)

    return results
