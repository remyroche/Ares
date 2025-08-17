# Enhanced validation wrapper
def enhanced_validate_features(
    data: pd.DataFrame, dataset_name: str = "features"
) -> Dict[str, Any]:
    """Enhanced validation with detailed logging"""

    from src.utils.data_quality_validator import validate_features
    import json
    from datetime import datetime

    # Run original validation
    results = validate_features(data, dataset_name)

    # Enhanced logging
    detailed_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "data_shape": data.shape,
        "total_features": len(data.columns),
        "validation_summary": results["summary"],
        "detailed_issues": {},
    }

    # Categorize issues by type
    issue_categories = {}
    for issue in results["issues"]:
        issue_type = issue.get("issue_type", "unknown")
        if issue_type not in issue_categories:
            issue_categories[issue_type] = []
        issue_categories[issue_type].append(issue)

    detailed_report["issue_categories"] = issue_categories

    # Feature-specific analysis
    feature_analysis = {}
    for col in data.columns:
        series = data[col]
        analysis = {
            "dtype": str(series.dtype),
            "missing_count": series.isna().sum(),
            "missing_percentage": (series.isna().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "most_common_value": series.mode().iloc[0]
            if len(series.mode()) > 0
            else None,
            "most_common_count": (series == series.mode().iloc[0]).sum()
            if len(series.mode()) > 0
            else 0,
        }

        if pd.api.types.is_numeric_dtype(series.dtype):
            analysis.update(
                {
                    "min_value": float(series.min()),
                    "max_value": float(series.max()),
                    "mean_value": float(series.mean()),
                    "variance": float(series.var()),
                    "infinite_count": float(np.isinf(series).sum()),
                    "extreme_count": float((series.abs() > 1e6).sum()),
                }
            )

        feature_analysis[col] = analysis

    detailed_report["feature_analysis"] = feature_analysis

    # Save detailed report
    report_file = f"validation_detailed_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(detailed_report, f, indent=2, default=str)

    print(f"📊 Detailed validation report saved to: {report_file}")

    return results


# Usage in step1_7_hmm_regime_discovery.py:
# Replace: validation_results = validate_features(features_df, f"features_{tf}")
# With: validation_results = enhanced_validate_features(features_df, f"features_{tf}")
