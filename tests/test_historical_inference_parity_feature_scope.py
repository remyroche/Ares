from scripts.historical_inference_parity import _feature_columns_for_state


class _DummyModel:
    def __init__(self, selected_features, input_feature_names=None):
        self.selected_features = selected_features
        if input_feature_names is not None:
            self.input_feature_names = input_feature_names


def test_historical_replay_uses_decision_feature_scope():
    state = {
        "bundle": {
            "alpha_models": {
                "long": {
                    "demo_head": {
                        "feat_cols": ["selected_alpha", "unused_union_alpha"],
                        "model": _DummyModel(["selected_alpha"]),
                    }
                }
            },
            "meta_models": {
                "long_demo_head": _DummyModel(["selected_meta"]),
            },
        }
    }

    keys = _feature_columns_for_state(state, "long_demo_head")

    assert "selected_alpha" in keys
    assert "selected_meta" in keys
    assert "unused_union_alpha" not in keys
