"""Run replication tasks in a separate copy; never overwrite archived results."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONSTANT = 'constant_cost'
TASKS = {
    'figure1': [('illustration/main.py', [])],
    'plots': [(f'{CONSTANT}/results/{name}.py', []) for name in
              ('plot_temp_horizon', 'plot_shift', 'plot_demand_horizon')]
             + [('changing_cost/plot_perf_time_scatter.py', [])],
    'timing-table': [('changing_cost/export_avg_time_by_ins_acc.py',
                      ['--dataset', 'regret', '--method', 'up', '--accuracy-func', str(1 + 25 * 10 / 49),
                       '--ins-acc', '1.02,1.05,1.08'])],
    'verification': [('verification/compare_three_methods.py',
                       ['--meshes', '0.0005', '--self-test', '--output', 'reproduced_results'])],
    'constant-tables': [(f'{CONSTANT}/{name}.py', []) for name in
                        ('generating-table-no-shift-execute',
                         'generating-table-no-shift-execute-perfect-only')],
    'horizon': [(f'{CONSTANT}/generating-execute-horizon.py', [])],
    'shift': [(f'{CONSTANT}/generate-shift-execute.py', [])],
    'demand-horizon': [(f'{CONSTANT}/demand-generate-execute-total-horizon.py', [])],
    'multi-phase': [('changing_cost/run_g_rr.py', [])],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=sorted(TASKS))
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory outside the source repository (must not exist).')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents or output in ROOT.parents:
        parser.error('Output must be outside the source repository and not its ancestor.')
    if output.exists():
        parser.error('Output already exists; choose a new path to preserve previous runs.')
    output.mkdir(parents=True)
    for folder in sorted({path.split('/')[0] for path, _ in TASKS[args.task]}):
        shutil.copytree(ROOT / folder, output / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '.venv', '.DS_Store'))
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(output / '.matplotlib'))
    for path, extra in TASKS[args.task]:
        script = output / path
        command = [sys.executable, str(script), *extra]
        print('Running:', ' '.join(command), flush=True)
        with (output / (script.stem + '.log')).open('w') as log:
            subprocess.run(command, cwd=script.parent, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed. Results and logs:', output)


if __name__ == '__main__':
    main()
