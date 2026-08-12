# Strict transport-gate audit: TP6/SL4 structural archetypes and GAM

Authoritative run:
`data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v5`

This supersedes the earlier K=6 diagnostic selection.  The earlier run was
useful for measuring the GAM, but it allowed a transport-insufficient cluster
partition to proceed.  This run puts structural transport and core/episodic
gates before economic cluster selection.

## Gate order

1. Build recurrent cross-fit archetypes.
2. Soft-match every leaf to at most three archetypes.
3. Retain explicit unmatched contribution mass.
4. Compute candidate-cluster transport metrics on development exposures only.
5. Require every cluster to be core or a genuinely separated episodic recurrence.
6. Apply balance/support gates.
7. Only then would the GAM be eligible for promotion.

The OOF GAM was still replayed diagnostically so that the comparison remains
available, but the manifest correctly marks the result as
`COMPLETE_DIAGNOSTIC_NO_VALID_CONTRACT` and
`economic_promotion_allowed=false`.

## Transport metrics

| Metric | Result |
|---|---:|
| Accepted archetypes | 11 |
| Weighted best-path similarity | 0.293 |
| Mean matched contribution mass | 0.514 |
| Mean unmatched contribution mass | 0.486 |
| Meta/context fields available to GAM | 313 eligible fields |
| Development rows | 6,816 |
| 2025 OOF rows | 10,224 |

Per-archetype and per-month metrics are in:

- `archetype_transport_detail.parquet`
- `archetype_transport_by_month.parquet`
- `cluster_transport_by_month.parquet`
- `selected_cluster_transport_gate.parquet`

Those artifacts include plausible-path fraction, maximum similarity,
contribution-weighted similarity, assigned mass, unmatched mass, activation,
conditional mass, membership entropy, recurrence, and core/episodic status.

## Candidate cluster gates

| K | Transport score | Transport coverage | Median mass coverage | Transport gate | Balance gate | Final valid |
|---:|---:|---:|---:|---|---|---|
| 2 | 0.891 | 1.000 | 0.262 | Pass | **Fail**: 72.9% mega-cluster | **No** |
| 3 | 0.616 | 0.792 | 0.064 | Pass | Fail | No |
| 4 | 0.601 | 0.750 | 0.069 | **Fail**: failed cluster | Pass | No |
| 5 | 0.629 | 0.800 | 0.075 | **Fail**: failed cluster | Pass | No |
| 6 | 0.627 | 0.771 | 0.084 | **Fail**: failed clusters | Pass | No |

The candidate ordering is intentionally transport-dominant.  K=2 transports
best, but its largest cluster contains 72.9% of mass, violating the
reasonable-balance gate.  K=3–6 produce partitions with at least one
contiguous, non-separated cluster that cannot be called a valid episodic
mechanism.  Therefore no economic cluster score is eligible for production.

## Selected diagnostic K=2 transport

| Diagnostic cluster | Dev coverage | Median mass | Mass CV | Status |
|---|---:|---:|---:|---|
| `cofire_cluster_00` | 100% | 40.1% | 0.168 | Core |
| `cofire_cluster_01` | 100% | 12.4% | 0.257 | Core |

The transport gate passes for this partition; the balance gate does not.

## GAM diagnostic result

The principal GAM remains:

```text
exposure × (beta + spline(context))
```

with prior-row CMI selection, train-only robust scaling, 3-knot splines,
membership as weight/exposure, and target:

```text
realised net_bps − base_expected_bps
```

The earlier diagnostic K=6 replay showed the representative zero-exposure
γ=.25 result:

| Tail | Base net | Diagnostic GAM net |
|---:|---:|---:|
| 0.5% | −98.12 bps | −79.34 bps |
| 1% | −106.56 bps | −108.17 bps |
| 2% | −32.76 bps | −28.93 bps |
| 5% | **+2.57 bps** | **+0.97 bps** |
| 10% | −8.94 bps | −8.03 bps |
| 20% | −29.54 bps | −28.13 bps |

Because K=6 fails the strict transport gate, these are diagnostic only and
must not be treated as an advancement.

## Decision

The required next step is to improve structural transport, not to tune the
GAM further.  In particular:

- reduce the roughly 49% unmatched contribution mass;
- use a learned/expanded path neighbourhood so recurring mechanisms are not
  missed by exact token overlap;
- retain a hard-top-1 activation channel alongside soft probabilities;
- require separated-era recurrence before allowing episodic clusters;
- rerun the cluster/GAM stage only after a valid balanced transport contract
  exists.
