# Stage-I target funnel: joint-stack shortlist and lineage contract

Round 3 is the target-shortlist decision point. It compares exactly three
finalists on the same chronological development holdout: frozen R3, the best
scalar S arm, and the best ordinal O arm. Ranking is always one pooled global
ranking after prior-resolved, side-local conversion to common expected-net bps.

Base-only economics are diagnostic. They order the family representatives and
record whether a challenger would have beaten the R3 base under the declared
tail/robustness checks, but they may not eliminate R3, S, or O. The immutable
`target_joint_shortlist_decision.json` is the **active** target-family
decision and therefore contains all three finalists and the complete base
diagnostic evidence. The older `target_promotion_decision.json` is
**legacy-only**: readers must not use it as an active promotion decision or a
Stage-I → Stage-II handoff.

`winner_bundles/joint_finalists/target_finalist_contracts.json` is the only
handoff permitted into target-specific work. It binds the existing R3 base and
the best scalar-S and ordinal-O bundles. Each finalist must receive its
matching direct three-class correctness meta layer. The terminal target choice
is then made only from reconstructed base+meta scores on identical rows after
causal common-bps mapping; no base-only result can promote or reject a target.

For a three-family comparison, the same contract additionally binds the signed
`stage_i_joint_finalist_shared_population` artifact. It is the side-qualified
intersection of target-valid R3/scalar/ordinal rows, applied before final OOS
fitting and mapping. A comparator rejects any multi-finalist ledger that does
not carry that exact shared-population hash.

This is development target selection, not an OOS or production promotion. The
separate 2024–2026 validation and execution-readiness gates remain required.

## Target labels and source invalidation

New exact-H12 path packs are schema v2. They bind the source-code closure for
the entry, ATR and path implementation, plus an inventory of every overlapping
minute fragment used for each symbol/time range. The fragment inventory hashes
contents, not only file names or modification times. A `--resume` succeeds
only if the exact request, selector inputs, source code and minute inventory
all match, and every published artifact still hashes correctly.

New target grids are schema v2 and carry forward the path-pack source-code and
minute-inventory contracts together with their own materializer source hashes.
They likewise fail closed on any `--resume` drift. This prevents later minute
repairs, loader changes, or label-code changes from being mistaken for the
same target surface.
