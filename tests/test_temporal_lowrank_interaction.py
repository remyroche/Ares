from __future__ import annotations

import torch

from scripts.run_sparse_parent_temporal_cnn_ablation import CausalTemporalResidualNet


def test_lowrank_mechanism_archetype_branch_is_factorized() -> None:
    torch.manual_seed(7)
    model = CausalTemporalResidualNet(
        channels=4,
        static_dim=10,
        architecture="mlp_lowrank",
        dropout=0.0,
        widths=(16, 12, 8),
        lookback_bars=16,
        archetype_dim=2,
        mechanism_positions=(2, 3, 4),
    ).eval()
    sequence = torch.zeros((2, 4, 16))
    static = torch.zeros((2, 10))
    static[:, 2:5] = torch.tensor([0.8, 0.2, 0.5])
    static[0, -2] = 1.0
    static[1, -1] = 1.0
    output = model(sequence, static)
    assert output.shape == (2,)
    assert model.mechanism_projection is not None
    assert model.archetype_projection is not None
    assert not torch.allclose(output[0], output[1])


def test_plain_model_does_not_create_lowrank_branch() -> None:
    model = CausalTemporalResidualNet(4, 10, "mlp", 0.1, (16, 12, 8), 16)
    assert model.mechanism_projection is None
    assert model.archetype_projection is None
