from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_orchestrator_is_research_only_and_guards_all_panels() -> None:
    source = (ROOT / "scripts" / "orchestrate_strict_r3_p8u_uom_cmi_prescreens_v1.py").read_text()
    for token in (
        '"2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"',
        "exact target-free monthly panels",
        "no MC1/admission/portfolio/live/execution/exchange mutation",
        "target_free",
        "base_identity_matched",
        "--top-base-fraction", '"0.15"',
        "mda_requested",
    ):
        assert token in source
