# Minimax-MDP numerical experiments

Code and data for the numerical experiments in **A Minimax-MDP Framework with
Future-imposed Conditions for Learning-augmented Problems**.

## Where to start

- [PDF figure/table map](docs/EXPERIMENTS.md): exact figure and table numbers, PDF pages, scripts and output files for the supplied paper PDF.
- [Running instructions](docs/REPRODUCING_PAPER.md): installation, commands and output locations.
- [Validation summary](docs/VALIDATION.md): checks performed on the code and stored results.

## Directory structure

| Directory | Contents |
|---|---|
| `illustration/` | Section 5.3: price-to-cost illustration |
| `verification/` | EC.7.2: closed-form, LP and DP comparison |
| `constant_cost/` | EC.7.2: constant-cost ordering experiments |
| `changing_cost/` | EC.7.3: changing-cost ordering experiments |
| `scripts/` | Reproduction, plotting, table export and packaging entry points |
| `docs/` | Instructions and run reports |

Input datasets and reference outputs are stored under each experiment's
`results/` or `result/` directory.

The Git repository stores five large input datasets in compressed form under
`data/inputs/`. The one-command workflow restores them automatically (about
1.6 GB of free disk space is needed). If running an individual experiment
script, restore the inputs first with `python scripts/restore_inputs.py`.

## Install and run

Python 3.12 is recommended. From this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python scripts/reproduce_paper.py --workers 4 --output ../paper-results
```

Choose a new output directory. The command recomputes performance from the
provided input datasets, compares it with stored results, and generates figures
and tables. It does not overwrite the inputs or reference outputs.

Historical runtime plots use the supplied timing measurements; new runs on
different hardware need not produce the same seconds. Optional numerical
accelerators are used for result verification, not historical timing claims.
The detailed guide also explains individual experiment commands.

## Package the code and data

```sh
python scripts/package.py --output ../minimax-MDP-code.zip
```

The ZIP includes the restored input files, a checksum manifest, and
instructions. The compressed Git copies, Git metadata, and Python caches are
excluded. Restore the inputs before packaging a fresh Git clone.
