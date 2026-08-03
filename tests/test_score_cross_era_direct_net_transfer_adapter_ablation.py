from scripts.score_cross_era_direct_net_transfer_adapter_ablation import parser


def test_label_free_scorer_has_no_current_label_argument():
    names = {action.dest for action in parser()._actions}
    assert "current_labels" not in names
    assert {"source_dir", "current_pack", "output_dir"}.issubset(names)
