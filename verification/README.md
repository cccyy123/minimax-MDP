# Verification of the closed-form robust values

This folder reproduces the five-instance comparison in Section EC.7.2, Tables EC.1 and EC.2. It compares **Closed form**, **LP** (the single-switch LP), and **DP** for robust regret and competitive ratio. Each objective is optimized separately.

## Contents

- `compare_three_methods.py`: self-contained implementation, five input instances, and small-instance checks.
- `requirements.txt`: pinned dependencies, NumPy 2.2.6 and SciPy 1.15.3.
- `three_method_results/comparison.md`: saved parameter table, comparison table, and grid sensitivity results.
- `three_method_results/results.json`: saved numerical results, LP certificates, discretization bounds, software versions, and the source script's SHA-256 hash.

The script and saved results are the versions used for the manuscript comparison. The command recorded in the saved JSON uses the script's original directory name, `solver_validation`; use the commands below in this repository.

## Reproduce

Python 3.12 is recommended; the saved results used Python 3.12.14. From the repository root:

```bash
cd verification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python compare_three_methods.py --meshes 0.0005 --self-test --output reproduced_results
```

This reproduces the grid spacing used in Table EC.2, delta = 0.0005, and leaves the saved reference results untouched. To also reproduce the grid sensitivity comparison:

```bash
python compare_three_methods.py --meshes 0.005 0.001 0.0005 --self-test --output reproduced_results
```

Those three meshes are also the defaults. The default output directory is `three_method_results`; using `--output reproduced_results` preserves the reference files.

To run only the small checks without the full experiments:

```bash
python -c "from compare_three_methods import self_test; self_test()"
```

No commercial solver or random sampling is required. Fine-grid DP uses about 1.54 GB for the largest main array, plus work buffers. The initial interval width, prediction widths, and capacities must be integer multiples of the selected mesh. `--max-cells` defaults to 250,000,000 elements. Edit `examples()` to change the instances.

## Instances and methods

The five horizons are T = 1, 3, 4, 6, and 10. The second instance is the example in Section EC.3.4. One instance is a single-period control; the other four were deliberately selected to exhibit a competitive-ratio gap between the full game and the single-switch restriction. They are not a random sample. Full parameters are in `examples()` and the saved comparison table.

- **Closed form:** evaluate the expressions in Propositions 2 and 3 using rational arithmetic.
- **LP:** optimize over the T+1 single-switch prediction sequences defined in Section EC.3.4. Decisions with the same history share variables. Orders remain continuous, and terminal demand is checked at both interval endpoints. The LP uses SciPy's HiGHS dual simplex solver.
- **DP:** backward induction on discrete prediction endpoints and cumulative inventory, following the state/action formulation of Section EC.7.4. Sliding extrema enumerate the grid transitions efficiently. The DP does not use the closed-form or LP values to compute its answers.

The single-switch LP ratio is the optimum against a **restricted adversary**, not a full-game guarantee. In Instances II--V it exceeds the closed-form ratio, consistently with the limitation illustrated in EC.3.4. This LP uses the paper's profit-based objective, rather than the imbalance-cost objective in the staffing model.

The implementation uses direct bounds on grid shifts (`k * delta`), with no extra flooring of that product. Equal consecutive prediction widths permit only zero shift. Both regret and competitive-ratio recursions and their terminal payoffs are implemented explicitly.

## Checks and numerical accuracy

`--self-test` runs 40 comparisons against exact rational DP on small games, including shorter prediction intervals and singleton intervals. It also checks analytical LP values, the EC.3.4 example, zero-shift transitions, and rejection of nonaligned meshes. Failures raise errors.

LP primal and dual feasibility tolerances are both 1e-9. The script additionally checks rational primal/dual certificates, with a certificate gap at most 1e-7. DP uses float64 backward induction; its small-game comparison tolerance is 1e-10. This arithmetic comparison tolerance is distinct from the error caused by discretization.

Let G_delta and F_delta denote the exact-arithmetic grid optima for regret and competitive ratio, respectively, let l be the initial demand lower bound, and let L = max(c, p-c). Under the script's grid-alignment assumptions, the reported discretization envelopes are:

```text
G_delta - (p-c)*delta <= Gamma* <= G_delta + L*delta
max(0, F_delta)*l/(l+delta) <= Phi* <= min(1, F_delta + delta/l)
```

These bounds account for endpoint and order discretization, but do not include rigorous floating-point interval bounds. They follow by rounding demand endpoints and cumulative orders; the cumulative order-rounding error is less than one mesh step. The script checks that the closed-form values lie in these envelopes, allowing 1e-10 floating-point slack.

At delta = 0.0005, the saved maximum absolute differences from the closed-form values are approximately:

| Comparison | Maximum absolute difference |
|---|---:|
| LP regret | 3.33e-16 |
| DP regret | 1.67e-4 |
| DP competitive ratio | 9.53e-5 |

Running time and floating-point trailing digits may vary by machine. The recorded SHA-256 hash verifies that the saved results correspond to the included source script.
