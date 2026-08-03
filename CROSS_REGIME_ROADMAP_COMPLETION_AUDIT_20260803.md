# Cross-Regime Roadmap Completion Audit

Audit date: 2026-08-03

Authoritative specification:
`/Users/remyroche/.codex/attachments/1de5ac97-5237-4d2d-82b9-8770ca1be1a4/pasted-text-1.txt`

Authoritative Stage-III architecture override:
`/Users/remyroche/.codex/attachments/8630da8d-21db-4a3e-8791-5a087b72c035/pasted-text.txt`

## Executive decision

The roadmap is **implemented structurally but not completed experimentally**.
The code and contract tests provide strong evidence that the intended causal,
identity, feature-lineage, pooled-ranking, and release boundaries exist. They
do not provide the winners or the requested 2024--2026 economics.

No experiment is currently running. The last Stage-I selector was stopped at
the user's request. Restarting training, HPO, materialisation, replay, or OOS
scoring still requires explicit authorization.

## Requirement-to-evidence matrix

| Stage | Required outcome | Current evidence | Decision |
|---|---|---|---|
| I | Per-side, per-layer, active-head feature selection using declared `config.py` pools | `stage_i_feature_selection.py` resolves only the base or meta keys, enforces the four active side/layer cells, and requires the direct same-side base OOF handoff for meta | Implemented |
| I | Coverage, univariate, Relief, 0.95 correlation pruning, grouped repeated signed-economic MDA, phantom/noise threshold, automatic feature count | `stage_i_feature_selection.py` and the selector implement the ordered screen and one-standard-error prefix choice. Dedicated MDA uses three disjoint chronological cohorts, <=20K training rows per model and up to 60K unique aggregate support, with overlap hashes plus worst/latest-era stability fields | Implemented/tested |
| I | Efficient/restartable large selection | Deterministic 32-column PIT checkpoints, schema/identity validation, invalidation scoped to the affected block, readiness-aware >=90% coverage gates, explicit causal warm-up boundaries, and cleanup only after the combined manifest is durable | Implemented; 80K selector materialisation active |
| I | Strict OOS predictions over 2024--2026, base and meta, per side/month, with/without causal 21-day admission | `stage_i_production_oos.py` freezes four winners, preserves separate base/meta OOF ledgers, and emits pooled-global tails plus unchanged-set attribution | Infrastructure only; no winner or economic output |
| II | Meta-specific archetype discovery and strict-OOF causal recognition | `stage_ii_meta_archetypes.py` and `stage_ii_meta_archetype_funnel.py` implement the bounded candidate funnel and four matched meta-only controls | Implemented |
| II | Frozen winner and untouched OOS evaluation | `stage_ii_production_oos.py` binds the winner, features, base artifact, windows, identities, scorer model, and mapping lineage; reselection/HPO are forbidden in evaluation | Implemented; no locked OOS run |
| III | One shared regime-aware residual expert, not local experts | `stage_iii_shared_expert_runner.py`, the preregistered JSON contract, and supporting modules implement A/T/B/C/D/E/F with one both-side model and reject hard/local routing | Implemented |
| III | Cross-era feature transport, robust objectives, causal regime/trust conditioning, calibration, pairwise tail loss | Dedicated Stage-III admission, robust-target, calibration, pairwise, reporting, and artifact modules cover these contracts | Implemented; no economic arm run |
| IV | Broad base then tail base; x=20/30/40/50; independent burn-ins; broad-score routing; row-level same-side OOF meta input | `stage_iv_broad_to_tail.py` and `stage_iv_v_orchestration.py` require explicit serial cells and compare them on the common strict-OOF identity intersection | Implemented; no cell run |
| V | MDA co-firing and compact drift/OOD controller, per side/layer, no reranking | `stage_v_drift_ood.py` and the Stage-IV/V orchestrator freeze training-only controller contracts and alter only permitted model inputs | Implemented; no controller arm run |
| VI | Separate causal and path archetypes, positive-label discovery, side-local fits, CF/PF views, K grid, AW weights, PCA/AE-GMM, strict-OOF path recognition | `stage_vi_archetypes.py` implements the bounded grids, soft memberships, causal recognizer, calibration/economic-confusion diagnostics, semantic alignment, and matched control/base/meta/both comparison | Implemented; no arm run |
| I--VI | Pooled-global ranking after common-bps mapping, never per timestamp/side/month | Stage-specific reporters select the global tail once and attribute the identical selected identities afterward | Implemented/tested |
| I--VI | Detailed base/meta and admission economics by side/month/week and worst period | Production reporters expose the required schemas and gates | No requested 2024--2026 result exists |
| I--VI | Thorough experiment report: trials, winners, failures, reasons | `CROSS_REGIME_SEQUENTIAL_ABLATION_REPORT_20260803.md` accurately separates implementation evidence from economic evidence | Active/incomplete until experiments run |

## Verified substrates already retained

- 2024 reference surface: January--November, 920,460 valid rows, 460,230 per
  side. December is absent from the frozen candidate source and remains an
  explicit coverage gap.
- August 2022--December 2023 exact-label history: 118,734 rows, 110,813 valid.
- Pack-B audit: 4,515,650 rows from January 2025--July 2026; the full 181-symbol
  universe has severe minute-path gaps in January--April 2026.
- Bounded common-30 repair: 5,230,800/5,230,800 required January--April minute
  observations verified against the frozen product identity, with
  172,800/172,800 regenerated exact-label rows valid.

These substrates are evidence of input readiness only. They are not evidence
that a model or feature-selection arm succeeds economically.

## Structural verification

The integrated Stage-I--VI contract suite most recently passed 180 tests. The
covered properties include:

- exact +1-hour decision and +13-hour label-availability timing;
- strict side-local OOF base-to-meta handoff;
- direct R3 simplex plus prequential expected-net map;
- separate base and meta OOF availability;
- exact candidate identity and content digests;
- feature coverage/non-constant gates;
- cost applied once and common-bps reconstruction;
- causal, side-local 21-day admission provenance;
- pooled-global selection and contribution-only attribution;
- immutable winner/release artifacts;
- no local Stage-III experts or hard regime routing;
- strict-OOF path-archetype recognition and no realised-path inference leakage.

This verification is necessary but deliberately insufficient for promotion.

## Exact remaining execution sequence

The experiment must resume at Stage I; later stages cannot select a valid
winner without the frozen predecessor.

1. Restart the checkpointed four-cell Stage-I selector.
2. Freeze base-long, base-short, meta-long, and meta-short winners, including
   ordered features, parameters, data hashes, and the approved historical
   feature-selection exception.
3. Generate the Stage-I 2024--2026 strict OOS ledgers and raw/admitted report.
4. Run Stage-II development selection, freeze its winner, then score the locked
   OOS population once.
5. Run Stage III sequentially through A, target, B, C, D, E, and F; stop at any
   declared advancement gate that fails.
6. If Stage III advances, run the explicit Stage-IV broad-to-tail cells on the
   common strict-OOF population.
7. If Stage IV advances, run Stage-V compact drift/OOD controller arms.
8. If Stage V advances, run the separate Stage-VI causal, path, and multi-view
   archetype workstreams.
9. Update the cumulative report after every completed trial with parameters,
   selected features, windows, row counts, predictive metrics, top-tail gross
   and net economics, causal admission results, side/month/week attribution,
   worst period, concentration, uncertainty, and disposition.

## Completion gate

The roadmap is complete only when the ordered experiment has either:

- produced and frozen a valid winner and requested OOS report at every stage
  that passes its advancement gate; or
- terminated at a predeclared gate with the failing trial evidence, the exact
  terminal decision, and no later dependent stage incorrectly executed.

At present neither terminal condition is satisfied because Stage I has no
selected winner and no full requested OOS economic result.
