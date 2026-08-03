from pathlib import Path


def test_bocpd_runner_has_three_explicit_resumable_modes():
    source = Path("scripts/run_strict_bocpd_regime_transition_challenger.py").read_text()
    assert 'choices=("context", "head", "seal")' in source
    assert 'context requires --horizon and --split' in source and 'parser.add_argument("--signal", choices=CHANGEPOINT_INPUT_COLUMNS)' in source
    assert 'head requires exactly one --head' in source
    assert 'context checkpoint per signal×horizon×split' in source


def test_context_is_per_signal_compact_hashed_and_uses_the_tested_primitive():
    source = Path("scripts/run_strict_bocpd_regime_transition_challenger.py").read_text()
    assert "bocpd_student_t_run_summary(values[run], config)" in source
    assert 'folder / f"{signal}.parquet"' in source
    assert 'folder / f"{name}.json"' in source
    assert '"context_sha256": sha256(path)' in source
    assert '"model_sample_cadence": "1h"' in source


def test_head_is_isolated_and_seal_is_merge_only():
    source = Path("scripts/run_strict_bocpd_regime_transition_challenger.py").read_text()
    assert "def run_head(" in source and "if head not in targets" in source
    assert "def seal(" in source
    assert "cannot seal before one-head bundles exist" in source
    assert "one pooled global top10 per UTC month" in source


def test_seal_uses_the_head_column_not_the_dataframe_method():
    source = Path("scripts/run_strict_bocpd_regime_transition_challenger.py").read_text()
    assert 'forward["head"].eq(head)' in source
    assert 'forward["head"].eq("onset_h3")' in source
    assert "forward.head.eq(" not in source
