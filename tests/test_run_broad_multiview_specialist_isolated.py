from pathlib import Path


def test_isolated_runner_declares_fresh_process_per_fold():
    text=(Path(__file__).parents[1]/'scripts/run_broad_multiview_specialist_isolated.py').read_text()
    assert 'subprocess.run' in text
    assert "'--fold-index'" in text
    assert 'one fresh Python process per fold' in text
