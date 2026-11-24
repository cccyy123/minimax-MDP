# Code validation summary

The complete nine-step command in `scripts/reproduce_paper.py` has been run
successfully using Python 3.12, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.3
and Numba 0.61.2. The dependency snapshot is in `validation-environment.txt`.

| Check | Scope | Result |
|---|---|---|
| Closed-form / LP / DP comparison | Five instances, both objectives, DP mesh 0.0005 | Matches stored numerical results |
| Constant-cost tables | All 80 input samples in each dataset, including DP meshes 1 and 4 | Maximum non-timing statistic error below 2e-15 |
| Prediction shift | 101 groups × 80 samples × 3 policies × 2 objectives | 48,480 values checked; maximum error below 1e-13 |
| Demand/horizon | 40 groups × 100 samples × 3 policies × 2 objectives | 24,000 values checked; maximum error below 2e-13 |
| Changing costs | 150 configurations × 100 samples × 3 policies × 2 objectives | 90,000 values checked; maximum error below 2e-12 |
| Plot generation | Illustration, horizon, shift, demand/horizon, changing-cost scatter and frontier | Entry points completed |
| Table export | Stored statistics, phase counts and timing aggregation | CSV files generated |

The checks reuse supplied input datasets. Historical timings are reaggregated,
not asserted equal to measurements on a different machine. Plot rendering may
vary in fonts or margins across environments.

Optional accelerated recurrences were checked against the original loops:
the NumPy DP on 40 small inputs under both objectives and additional archived
cases; the compiled phase reduction on 60 cases with exact output agreement.
Changing-cost summation order is explicit to preserve historical arithmetic.
As a separate generator check, the first stored samples of the changing-cost,
noisy constant-cost, and consistent-and-nested constant-cost datasets were
regenerated from their recorded seeds and matched exactly. The full archived
random-input datasets were not regenerated in this check.

See `validation/summary.json` for the latest step completion, sample counts and
numerical errors. Running instructions
and output locations are in [REPRODUCING_PAPER.md](REPRODUCING_PAPER.md).
