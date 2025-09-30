from src.training.steps.market_analysis.regime_analysis import reporting


def test_print_detailed_metrics(monkeypatch):
    messages = []

    def fake_tprint(message, level):
        messages.append((message, level))

    monkeypatch.setattr(reporting, "tprint", fake_tprint)

    distribution = {
        "regime_counts": {"regime_0": 2},
        "regime_percentages": {"regime_0": 100.0},
        "regime_balance": {
            "min_percentage": 100.0,
            "max_percentage": 100.0,
            "std_percentage": 0.0,
            "balance_score": 1.0,
        },
    }
    metrics = {
        "silhouette_score": 0.5,
        "davies_bouldin_score": 0.2,
        "cv_score": 0.7,
        "interpretation": {
            "silhouette": "Good clustering",
            "davies_bouldin": "Excellent separation",
            "cv_score": "Good regime distinction",
        },
    }

    reporting.print_detailed_metrics(distribution, metrics, "NAS")

    assert any("NAS REGIME DETAILED ANALYSIS" in msg for msg, _ in messages)
    assert any("regime_0" in msg for msg, _ in messages)


def test_print_analysis_summary(monkeypatch):
    messages = []

    def fake_tprint(message, level):
        messages.append((message, level))

    monkeypatch.setattr(reporting, "tprint", fake_tprint)

    analysis = {
        "nas_analysis": {
            "distribution": {
                "num_regimes": 1,
                "total_samples": 2,
                "regime_balance": {
                    "min_percentage": 50.0,
                    "max_percentage": 50.0,
                    "std_percentage": 0.0,
                    "balance_score": 1.0,
                },
            },
            "clustering_metrics": {
                "silhouette_score": 0.5,
                "davies_bouldin_score": 0.2,
                "cv_score": 0.7,
                "interpretation": {
                    "silhouette": "Good clustering",
                    "davies_bouldin": "Excellent separation",
                    "cv_score": "Good regime distinction",
                },
            },
        },
        "tas_analysis": {
            "distribution": {
                "num_regimes": 1,
                "total_samples": 2,
                "regime_balance": {
                    "min_percentage": 50.0,
                    "max_percentage": 50.0,
                    "std_percentage": 0.0,
                    "balance_score": 1.0,
                },
            },
            "clustering_metrics": {
                "silhouette_score": 0.5,
                "davies_bouldin_score": 0.2,
                "cv_score": 0.7,
                "interpretation": {
                    "silhouette": "Good clustering",
                    "davies_bouldin": "Excellent separation",
                    "cv_score": "Good regime distinction",
                },
            },
        },
    }

    reporting.print_analysis_summary(analysis)

    assert any("REGIME ANALYSIS SUMMARY" in msg for msg, _ in messages)
    assert any("NAS REGIMES" in msg for msg, _ in messages)
    assert any("TAS REGIMES" in msg for msg, _ in messages)
