from scripts.materialize_capture_feature_universe import prefixed_feature_names


def test_capture_feature_universe_prefix_is_collision_safe() -> None:
    assert prefixed_feature_names(["atr", "volume"]) == [
        "capture_candidate__atr",
        "capture_candidate__volume",
    ]
