# Paper PDF to code map

This guide maps the numerical results in the revised manuscript to code and
outputs. PDF page numbers below count from the first page of the current
compiled manuscript; printed appendix page labels are also included.

Run the following from the code root after installing the dependencies described
in [REPRODUCING_PAPER.md](REPRODUCING_PAPER.md):

```sh
python scripts/reproduce_paper.py --workers 4 --output ../paper-results
```

All output paths in the next table are relative to `../paper-results/`.
Choose an output directory that does not already exist.

## Figures

Each figure has regret on the left and competitive ratio on the right. The code
saves these panels as separate PNG files.

| PDF figure | Section; printed page (PDF page) | Computation / plotting code | Output panels, left then right |
|---|---|---|---|
| Figure 1: price-to-cost ratio | 5.3; 27 (27) | `illustration/main.py` | `figure1/illustration/figure_regret.png`; `figure1/illustration/figure_ratio.png` |
| Figure EC.2: running time versus horizon | EC.7.2; ec69 (105) | `constant_cost/generating-execute-horizon.py`; `constant_cost/results/plot_temp_horizon.py` | `figures/constant_cost/results/picture/result_rgt_horizon_time.png`; `result_ratio_horizon_time.png` in the same directory |
| Figure EC.3: biased prediction intervals | EC.7.2; ec70 (106) | `constant_cost/generate-shift-execute.py`; `scripts/replay_prediction_figures.py`; `constant_cost/results/plot_shift.py` | `figures/constant_cost/results/picture/result_rgt_shift.png`; `result_ratio_shift.png` in the same directory |
| Figure EC.4: variability versus horizon | EC.7.2; ec70 (106) | `constant_cost/demand-generate-execute-total-horizon.py`; `scripts/replay_prediction_figures.py`; `constant_cost/results/plot_demand_horizon.py` | `figures/constant_cost/results/picture/result_rgt_demand_horizon.png`; `result_ratio_demand_horizon.png` in the same directory |
| Figure EC.5: changing-cost performance/time scatter | EC.7.3; ec72 (108) | `changing_cost/g.py`, `rr.py`, `phase_reduction.py`; `scripts/replay_changing_cost_all.py`; `changing_cost/plot_perf_time_scatter.py` | `figures/changing_cost/result/scatter_regret_time_log_vs_performance.png`; `scatter_ratio_time_log_vs_performance.png` in the same directory |
| Figure EC.6: changing-cost Pareto frontier | EC.7.3; ec72 (108) | Same computation as Figure EC.5; non-dominated filtering in `changing_cost/plot_perf_time_scatter.py` | `figures/changing_cost/result/scatter_regret_time_log_vs_performance_filtered.png`; `scatter_ratio_time_log_vs_performance_filtered.png` in the same directory |

Figure EC.2 uses supplied historical timing measurements. Figures EC.5–EC.6
combine recomputed performance with historical timings. Fonts and margins can
vary with the rendering environment; new hardware timings need not equal the
reported seconds.

## Tables

| PDF table | Section; printed page (PDF page) | Code and input | Output / how to read it |
|---|---|---|---|
| Table EC.3: five test instances | EC.7.2; ec67 (103) | `examples()` in `verification/compare_three_methods.py` | First table in `verification/verification/reproduced_results/comparison.md`; parameters also appear in `results.json` alongside it |
| Table EC.4: closed form, single-switch LP and DP | EC.7.2; ec67 (103) | `verification/compare_three_methods.py`, DP mesh `0.0005` | Second table in the same `comparison.md`, with both objectives; full precision in `results.json` |
| Table EC.5: consistent-and-nested predictions | EC.7.2; ec68 (104) | `constant_cost/generating-table-no-shift-execute-perfect-only.py`; `constant_cost/results/generate_table_perfect_only.json`; replay via `scripts/verify_tables.py` | Recomputed `tables/result_{ratio,rgt}_perfect_only.json`; reference statistics in `exported-tables/constant_cost_tables.csv`, selecting `dataset=_perfect_only` |
| Table EC.6: noisy predictions | EC.7.2; ec69 (105) | `constant_cost/generating-table-no-shift-execute.py`; `constant_cost/results/generate_table.json`; replay via `scripts/verify_tables.py` | Recomputed `tables/result_{ratio,rgt}.json`; reference statistics in `exported-tables/constant_cost_tables.csv`, selecting `dataset=noisy` |
| Table EC.7: phase counts, constraints and time | EC.7.3; ec73 (109) | `scripts/export_tables.py`; `changing_cost/export_avg_time_by_ins_acc.py`; archived changing-cost data | `exported-tables/phase_counts_and_times.csv`; phases `5,4,3`, constraints `33,17,9`, mean times rounded to `4.097,2.680,1.453` seconds |

Table EC.3 instances I–V correspond, in order, to `single_period`, `EC_3_4`,
`four_period`, `six_period`, and `ten_period` in the code. For Table EC.7 the
objective is regret, the policy is interval-max, and the demand-grid spacing is
`1 + 25*10/49` (approximately `6.1`).

For Tables EC.5–EC.6, JSON/CSV objective `rgt` means regret and `ratio` means
competitive ratio. Method keys `up`, `lw`, and `mid` mean interval-max,
interval-min, and interval-mid. Keys `dis_1`, `dis_2`, `naive_greedy`, and
`lp` correspond to dp-acc(1), dp-acc(4), naive-greedy, and LP-based. The CSV preserves full-precision reference
statistics, including historical timing; the replay JSONs contain newly computed
performance. The supplied table datasets contain 80 samples each, matching the current
Table EC.5 and EC.6 captions. The commands replay these fixed samples.

## Input files by figure

Paths below are relative to the code root. Braces abbreviate the two objective
files, rather than denoting a literal filename.

| Figure | Inputs / stored reference outputs |
|---|---|
| Figure 1 | Constants defined in `illustration/main.py`; no external input data |
| Figure EC.2 | `constant_cost/results/generate_horizon.json`; `result_{rgt,ratio}_horizon.json` and supplemental `result_rgt_horizon_lp.json` in that directory |
| Figure EC.3 | `constant_cost/results/generate_shift.json`; `result_{rgt,ratio}_shift.json` in that directory |
| Figure EC.4 | `constant_cost/results/generate_demand_horizon.json`; `result_{rgt,ratio}_demand_horizon.json` in that directory |
| Figures EC.5–EC.6 | `changing_cost/result/generate_decrease_K_multi_{regret,ratio}.json`; `result_decrease_K_multi_{regret,ratio}.json` in that directory |

Figures EC.5–EC.6 show all 150 rounded-cost configurations. The final 50
exact-cost reference configurations are excluded from these plots. See the
running guide for the sample-count conventions.

## Other PDF material

- EC.7.1 describes synthetic demand and prediction generation, implemented in
  the experiment generators above. Reproduction uses their saved input JSONs.
- EC.7.4 describes the constant-cost DP baseline used in Tables EC.4–EC.6 and
  Figure EC.2; it is not a separate experiment dataset.
- Figure EC.1 (ec17; PDF page 53) is a theoretical policy illustration.
  Tables EC.1 and EC.2 (ec27 and ec35; PDF pages 63 and 71) are notation tables.
  These are not numerical experiment outputs.
- `constant_cost/estimate-total-demand-interval-prob.py` and
  `constant_cost/results/result_totaldemand_interval_prob_table.json` support
  the prediction-coverage discussion; they do not correspond to a numbered
  experiment figure or table.

Individual run commands are in [REPRODUCING_PAPER.md](REPRODUCING_PAPER.md).
Completed numerical checks are summarized in [VALIDATION.md](VALIDATION.md).
