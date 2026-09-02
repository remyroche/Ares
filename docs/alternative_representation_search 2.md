# Alternative AE/GMM Representation Search

This experiment compares alternatives to the production sequential DAE/GMM
state representation without changing the production model or policy.

## Selection Contract

1. Build one immutable beginning/middle/end reference cache. Every 100k encoder
   comparison uses the same row indices and feature-value hash.
2. Fit the incumbent primary comparator on exactly the same 50k sampled rows as
   every alternative encoder and density panel.
3. Fit VaDE, two-stage IDEC, masked/SCARF/VICReg encoders, and frozen-embedding
   IIC heads. Encoder matrices, augmentations, donor maps, and posteriors are
   cached.
4. Rank every embedding with the same cheap GMM panel (`K=4,8`, diagonal,
   `reg_covar=0.003`). Promotion is outcome-free and family-relative.
5. Run the staged density search only for the top two embeddings per family.
   Latent standardization or whitening is serialized as part of the model.
6. Only frozen finalists enter the canonical supervised base and meta pipelines.
   Economic results never retune the outcome-free proxy weights.

The representation comparison is deliberately transductive: the outcome-free
reference sample spans beginning, middle, and end of the available covariate
period. Base/meta model fits and reported economic predictions remain
chronological. Reports must not describe representation discovery as untouched
OOS evidence.

## Stages

```bash
PYTHONPATH=. python3 scripts/run_alternative_representation_search.py \
  --labels-path <labels/dir> \
  --feature-dir <features/dir> \
  --output-root <report/dir> \
  --stage plan
```

The available stages are:

- `prepare`: immutable float32/memory-mapped reference matrices and hashes.
- `encoders`: incumbent, VaDE, IDEC proxy, and SSL encoder caches.
- `proxy`: common GMM panel, seed/resample/perturbation stability, entropy,
  occupancy by symbol/time/side/market regime, and three-tier OOD diagnostics.
- `idec_final`: expand only the two promoted IDEC proxy embeddings into a
  coverage-preserving final grid.
- `density1`: raw-latent diagonal GMM search on 100k rows.
- `density2`: promoted 250k raw/whitened diagonal/tied search.
- `overlap`: post-economic-screen bounded overlap refinement with held-out NLL
  checks. It is explicit and is not included in the initial `all` comparison.
- `iic`: post-economic-screen deterministic sample of the frozen-head IIC grid.
  It is explicit and is not included in the initial `all` comparison.
- `base`: run canonical global MDA once without any representation outputs,
  append each frozen candidate's outputs to that common feature contract, and
  reuse production base parameters. No per-candidate MDA or base HPO is run.
- `meta`: canonical Pack-B staged screening and independent side MDA, with the
  frozen production long/short parameters reused. Feature selection can admit
  candidate representation outputs, but no meta HPO is run.
- `report`: top-2 per-family overall/month/week/side/archetype economic metrics,
  including signed residual and hit-rate-surprise autocorrelation.

`all` executes the stages in this order. Expensive work is never started by the
default `plan` stage.

## Compute Controls

- `--max-candidates-per-family 12` is the default coverage-preserving proxy cap;
  use `0` for the complete VaDE/SSL grids. The cap greedily covers rare search
  axis values before filling the remaining budget.
- `--max-idec-final-candidates 12` limits the second IDEC stage.
- `--iic-trials-per-embedding 8` samples the otherwise very large IIC grid.
- SSL uses the same exact 50k comparison rows as the other alternative encoders. The
  reference transform retains 300k beginning/middle/end rows for temporal and
  OOD diagnostics.
- The common panel uses `K=4,8`, one initialization, and one seed.
  Density stage 1 tests five representative component counts; stage 2 tests
  `K=4,6,8`, raw versus whitened latent geometry, and diagonal versus tied
  covariance on at most six embeddings.
- Overlap refinement keeps both economic distance mechanisms but evaluates one
  density per family, four representative penalties, 12k fit rows, 30 update
  steps, and two seed rechecks.
- Base MDA uses 30k fit plus 10k chronological evaluation rows exactly once on
  representation-free features. Meta staged MDA uses 30k rows. Both retain
  beginning/middle/end sampling and archetype-aware scoring.
- IIC and overlap refinement are deferred until the first downstream economic
  screen identifies embeddings worth refining.
- Only the best density plus the best validated refinement and IIC arm per
  family reach base evaluation. The best base result per family reaches meta.
  The five-month OOS contract is unchanged.
- Base and meta comparisons reuse frozen production parameters. This removes
  Optuna variance and keeps the outer test focused on representation value.
- Neural arrays and latent caches use float32. GMM likelihood, covariance,
  whitening, and Mahalanobis operations use float64.
- The runner prefers MPS when PyTorch reports it available and otherwise falls
  back to CPU. Caching is the primary Mac optimization.

## Output Semantics

All mathematically valid outputs are exposed through a common prefix:

- `repr_latent_*`
- `repr_component_posterior_*`, entropy, margin, and Mahalanobis fields for a
  fitted density model
- native IDEC/VaDE posteriors where available
- reconstruction error
- VaDE posterior mean/variance and ELBO-derived novelty percentile
- density novelty raw score and frozen-reference percentile

`embedding_only` IDEC candidates intentionally omit categorical assignment
fields. Tied-covariance GMMs retain their shared-covariance Mahalanobis
calculation rather than being silently converted to diagonal inference.
