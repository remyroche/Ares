from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_objective_handoff_is_guarded_and_research_only() -> None:
    source = (ROOT / "scripts" / "orchestrate_strict_r3_p8u_uom_objective_heads_v1.py").read_text()
    for token in (
        '"under", "over", "magnitude"',
        '"top15"',
        '"candidate_support"',
        '"causal_receipt"',
        '"selection": "sealed full-1400 Base-Explanation-V1 top-15% conditional-MI prescreen; MDA intentionally not invoked"',
        'no MC1/admission/portfolio/live/execution/exchange mutation',
        '"--feature-count", "80"',
        '"--held-months", *HELD_MONTHS',
    ):
        assert token in source
