import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from scripts.materialize_early_2022_inverse_pi_regime_supplement import STATE_FEATURES, _component_alignment, _cross_lineage_bridge, _entropy


def test_entropy_is_bounded_and_bridge_rejects_taxonomy_alignment() -> None:
    values = _entropy(np.array([[0.5, 0.5], [1.0, 0.0]]))
    assert np.all(values >= 0.0) and np.all(values <= 1.0)
    assert not _cross_lineage_bridge().taxonomy_alignment_allowed.any()


def test_component_alignment_assigns_all_components() -> None:
    matrix = np.arange(8 * len(STATE_FEATURES), dtype=float).reshape(8, len(STATE_FEATURES))
    scaler = StandardScaler().fit(matrix); model = GaussianMixture(n_components=2, random_state=1).fit(scaler.transform(matrix))
    mapping, semantic, profile = _component_alignment(model, scaler)
    assert set(mapping.values()) == {0, 1}
    assert len(semantic) == len(profile) == 2
